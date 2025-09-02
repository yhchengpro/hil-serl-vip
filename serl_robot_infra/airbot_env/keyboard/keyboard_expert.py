from pynput import keyboard
import threading
import numpy as np

class KeyboardExpert:
    def __init__(self, speed=1):
        self.speed = speed
        self.lock = threading.Lock()
        self.delta_twist = np.zeros((2, 3))  # 线速度和角速度
        self.gripper = 0
        self.intervened = False
        self._start_listener()

    def _start_listener(self):
        listener = keyboard.Listener(on_press=self._on_press)
        listener.daemon = True
        listener.start()

    def _on_press(self, key):
        try:
            with self.lock:

                if hasattr(key, 'char'):
                    c = key.char.lower()
                    if c == 'w':
                        self.delta_twist[0][0] = +self.speed
                    elif c == 's':
                        self.delta_twist[0][0] = -self.speed
                    elif c == 'a':
                        self.delta_twist[0][1] = +self.speed
                    elif c == 'd':
                        self.delta_twist[0][1] = -self.speed
                    elif c == 'q':
                        self.delta_twist[0][2] = +self.speed
                    elif c == 'e':
                        self.delta_twist[0][2] = -self.speed
                    elif c == 'u':
                        self.delta_twist[1][0] = -self.speed * 30
                    elif c == 'o':
                        self.delta_twist[1][0] = +self.speed * 30
                    elif c == 'j':
                        self.delta_twist[1][1] = +self.speed * 30
                    elif c == 'l':
                        self.delta_twist[1][1] = -self.speed * 30
                    elif c == 'k':
                        self.delta_twist[1][2] = +self.speed * 30
                    elif c == 'i':
                        self.delta_twist[1][2] = -self.speed * 30
                    elif c == '[':
                        self.gripper = -1
                    elif c == ']':
                        self.gripper = 1
                elif key == keyboard.Key.enter:
                    self.intervened = not self.intervened
                
        except Exception:
            pass

    def get_action(self):
        with self.lock:
            # 返回动作并重置
            delta = self.delta_twist.copy()
            delta = [*delta[0], *delta[1]]
            grip = [self.gripper]
            self.delta_twist = np.zeros((2, 3))  # 每次重置增量
            self.gripper = 0
        return delta, grip, self.intervened
