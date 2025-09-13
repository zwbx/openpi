# multi_gpu_occupy.py
import time
import threading
import subprocess
import argparse
import torch

def get_util_percent(device=0):
    """
    用 nvidia-smi 读取指定 GPU 的利用率 (0-100)，失败返回 None。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits",
             "-i", str(device)],
            capture_output=True, text=True, check=True
        )
        util_str = out.stdout.strip().splitlines()[0]
        return int(util_str)
    except Exception:
        return None

def mem_used_mb(device=0):
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024**2)

def burn(device, stop_flag):
    """
    占卡负载线程：大矩阵乘法保持忙碌。stop_flag["stop"] 为 True 时退出。
    """
    torch.cuda.set_device(device)
    x = torch.randn((4096, 4096), device=f'cuda:{device}')
    y = torch.randn((4096, 4096), device=f'cuda:{device}')
    while not stop_flag["stop"]:
        z = torch.mm(x, y)
        x = z + 0.1
#

def someone_came_during_probe(device, grace_secs=3.0, util_thresh=50,
                              mem_thresh_mb=300, confirm_ticks=2, tick_interval=0.2):
    """
    让路后在窗口内观察：
    - 优先用 nvidia-smi 的 GPU 利用率 > util_thresh 判定；
    - 若读不到利用率，则退化为 显存占用 > mem_thresh_mb 判定；
    需要连续命中 confirm_ticks 次以去抖。
    """
    torch.cuda.set_device(device)
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    hits = 0
    t0 = time.time()
    while time.time() - t0 < grace_secs:
        util = get_util_percent(device)
        if util is not None:
            cond = (util > util_thresh)
        else:
            cond = (mem_used_mb(device) > mem_thresh_mb)

        if cond:
            hits += 1
            if hits >= confirm_ticks:
                return True
        else:
            hits = 0
        time.sleep(tick_interval)
    return False

def controller(device, probe_interval, grace_secs, util_thresh, mem_thresh_mb, confirm_ticks):
    """
    单卡控制线程：管理该 device 的占/让逻辑。
    """
    stop_flag = {"stop": False}
    worker = None
    last_probe = 0.0

    while True:
        if worker is None:
            # 进门前观察：有人→保持空闲；无人→启动占卡
            if someone_came_during_probe(device, grace_secs=grace_secs,
                                         util_thresh=util_thresh,
                                         mem_thresh_mb=mem_thresh_mb,
                                         confirm_ticks=confirm_ticks):
                print(f"[GPU{device}] 检测到他人占用，保持空闲")
            else:
                print(f"[GPU{device}] 空闲，开始占卡")
                stop_flag["stop"] = False
                worker = threading.Thread(target=burn, args=(device, stop_flag), daemon=True)
                worker.start()
                last_probe = time.time()
        else:
            # 周期性让路探测
            if time.time() - last_probe >= probe_interval:
                last_probe = time.time()
                print(f"[GPU{device}] 周期性让路探测…")
                stop_flag["stop"] = True
                worker.join(); worker = None
                torch.cuda.synchronize(); torch.cuda.empty_cache()

                if someone_came_during_probe(device, grace_secs=grace_secs,
                                             util_thresh=util_thresh,
                                             mem_thresh_mb=mem_thresh_mb,
                                             confirm_ticks=confirm_ticks):
                    print(f"[GPU{device}] 探测到新人接管，暂停占卡")
                else:
                    print(f"[GPU{device}] 无人使用，恢复占卡")
                    stop_flag = {"stop": False}
                    worker = threading.Thread(target=burn, args=(device, stop_flag), daemon=True)
                    worker.start()

        time.sleep(0.5)

def main():
    parser = argparse.ArgumentParser(description="多卡合作式占卡器")
    parser.add_argument("--devices", type=str, default="all",
                        help='逗号分隔的 GPU 序号，如 "0,2,3"；默认为 "all"')
    parser.add_argument("--probe_interval", type=int, default=25,
                        help="占卡期间每隔多少秒主动让路探测一次")
    parser.add_argument("--grace_secs", type=float, default=5,
                        help="让路观察窗口时长（秒）")
    parser.add_argument("--util_thresh", type=int, default=5,
                        help="利用率阈值（%），超过则判定有人用")
    parser.add_argument("--mem_thresh_mb", type=int, default=300,
                        help="没有利用率时的显存占用阈值（MB）")
    parser.add_argument("--confirm_ticks", type=int, default=1,
                        help="观察窗口内需连续命中次数以去抖")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("没有可用的 GPU"); return

    total = torch.cuda.device_count()
    if args.devices == "all":
        devices = list(range(total))
    else:
        devices = [int(d.strip()) for d in args.devices.split(",") if d.strip() != ""]

    print(f"管理的 GPU: {devices}")

    threads = []
    for d in devices:
        t = threading.Thread(
            target=controller,
            args=(d, args.probe_interval, args.grace_secs, args.util_thresh, args.mem_thresh_mb, args.confirm_ticks),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断，退出。")

if __name__ == "__main__":
    main()
