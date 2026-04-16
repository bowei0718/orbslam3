import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS
import pygame
import os

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        # 訂閱來自 LLM 的回應文字
        self.subscription = self.create_subscription(
            String, 'ai_response', self.listener_callback, 10)
        pygame.mixer.init()

    def listener_callback(self, msg):
        text = msg.data
        self.get_logger().info(f'📢 準備播報: "{text}"')
        
        # 將文字轉為語音檔案
        tts = gTTS(text=text, lang='zh-tw')
        tts.save("response.mp3")
        
        # 播放
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()