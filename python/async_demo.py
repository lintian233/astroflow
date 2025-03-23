import multiprocessing
import time
import random


class Task:
    def __init__(self, task_id):
        self.task_id = task_id

    def execute(self):
        # 模拟任务执行
        time.sleep(random.uniform(1.0, 3.0))
        print(f"Task {self.task_id} completed")


class TaskManager:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.waiting_queue = multiprocessing.Queue()
        self.in_progress_queue = multiprocessing.Queue()
        self.queue_lock = multiprocessing.Lock()  # 新增进程锁
        self.workers = []
        self._running = False

    def _worker(self):
        while True:
            with self.queue_lock:  # 加锁获取任务
                if not self.waiting_queue.empty():
                    task = self.waiting_queue.get()
                else:
                    task = None

            if task is None:
                if not self._running:  # 双重检查退出条件
                    break
                time.sleep(0.1)
                continue

            with self.queue_lock:  # 加锁更新进行队列
                self.in_progress_queue.put(task)
                print(f"Task {task.task_id} started")

            task.execute()

            with self.queue_lock:  # 加锁移除任务
                self.in_progress_queue.get()
                print(f"Task {task.task_id} removed from in-progress queue")

    def start(self):
        """启动工作进程池"""
        if not self._running:
            for _ in range(self.num_workers):
                p = multiprocessing.Process(target=self._worker)
                p.start()
                self.workers.append(p)
            self._running = True

    def submit(self, task):
        """提交任务到队列并自动启动工作进程"""
        if not self._running:
            self.start()
        self.waiting_queue.put(task)
        print(f"Task {task.task_id} added to waiting queue")

    def shutdown(self):
        """安全关闭工作进程"""
        self._running = False

        # 安全清空队列
        with self.queue_lock:
            while not self.waiting_queue.empty():
                self.waiting_queue.get()
            while not self.in_progress_queue.empty():
                self.in_progress_queue.get()

        # 等待工作进程结束
        for p in self.workers:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
        self._running = False

    def join(self):
        while True:
            with self.queue_lock:
                if self.waiting_queue.empty() and self.in_progress_queue.empty():
                    break
            time.sleep(1)

        # 等待所有工作进程自然退出
        self.shutdown()


if __name__ == "__main__":
    # 使用示例
    manager = TaskManager(num_workers=32)

    for i in range(10):
        task = Task(i)
        manager.submit(task)

    # 安全关闭
    manager.join()
