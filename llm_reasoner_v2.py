import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from google import genai

# 標準配置
API_KEY = "AIzaSyCq3XCt1QaOHjkLfaYaqOtzNFwYYMftmnA"
client = genai.Client(api_key=API_KEY)
CURRENT_MODEL = 'gemini-3-flash-preview' 

class LLMReasonerNode(Node):
    def __init__(self):
        super().__init__('llm_reasoner_node')

        # 1. 建立對話記憶列表
        self.chat_history = []
        
        # 讀取地圖資訊
        json_path = "/home/asrlab215/colcon_ws/semantic_locations_graph_0413_v3.json"
        with open(json_path, 'r') as f:
            self.map_data = json.load(f)

        # 訂閱語音辨識結果
        self.subscription = self.create_subscription(
            String, '/voice_text', self.command_callback, 10)

        # 原有的目標發布 (導航用)
        self.publisher_ = self.create_publisher(String, '/target_id', 10)

        # 新增：發布語音回覆 (TTS 用)
        self.tts_publisher = self.create_publisher(String, '/ai_response', 10)

        self.get_logger().info('🧠 The semantic reasoning brain (including angle determination) has been activated.')

    def build_world_description(self):
        """將 JSON 轉成環境描述"""
        desc = "【環狀走廊語義地圖】\n物件 ID 採『逆時針排序』，door_1 離起點最近。\n"
        for category, items in self.map_data.items():
            for item in items:
                # 明確告知每個 ID 對應的房間，方便 AI 判斷方向
                desc += f"- {item['instance_name']}: 位於 {item['corridor_id']}, 連接房間 {item['room_id']}\n"
        return desc

    def command_callback(self, msg):
        user_command = msg.data
        self.get_logger().info(f'[### Speech Recieve] Instruction received: "{user_command}"')

        # 將方向對應表寫入 Prompt
        system_context = f"""
        System Context：{self.build_world_description()}
        
        You are a robot assistant. Please answer in Chinese.

        Navigation Orientation Rules (Yaw Degrees)

        Base orientation rules are defined for corridor_1.
        If the target object is connected to another corridor, the orientation rules must be rotated counter-clockwise by 90 degrees for each corridor level.

        Corridor priority rule (higher priority overrides lower):
        corridor_3 > corridor_2 > corridor_1 > corridor_4

        Corridor rotation logic:
        - corridor_1 → base rules
        - corridor_2 → corridor_1 rotated CCW 90°
        - corridor_3 → corridor_2 rotated CCW 90°
        - corridor_4 → corridor_3 rotated CCW 90°

        If multiple corridors appear in the object's connection graph:
        use the highest priority corridor rule.

        --------------------------------------------------

        Corridor_1 Navigation Orientation Rules (Base):

        - Right (Enter room/lab on the right)：180
        - Left (Enter room/lab on the left)：0
        - Forward (Move up along the corridor)：-90
        - Backward (Move down along the corridor)：90

        Default angle if no direction specified: -90

        --------------------------------------------------

        Corridor_2 Rules (corridor_1 rotated CCW 90°):

        - Right：-90
        - Left：90
        - Forward：0
        - Backward：180

        Default angle: 0

        --------------------------------------------------

        Corridor_3 Rules (corridor_2 rotated CCW 90°):

        - Right：0
        - Left：180
        - Forward：90
        - Backward：-90

        Default angle: 90

        --------------------------------------------------

        Corridor_4 Rules (corridor_3 rotated CCW 90°):

        - Right：90
        - Left：-90
        - Forward：180
        - Backward：0

        Default angle: 180

        --------------------------------------------------

        Task:

        1. Identify the "instance_name" that best matches the user command.

        2. Determine which corridor rule should be applied:
        - If the object connection contains corridor_2 → use corridor_2 rules.
        - If it contains corridor_3 → use corridor_3 rules.
        - If it contains corridor_4 → use corridor_4 rules.
        - Otherwise use corridor_1 rules.

        3. Determine the target yaw angle based on the selected corridor rule.

        4. AFTER the navigation code, provide a natural language response for the user to hear.

        5. If the user is just chatting or asking a question, only provide a natural language response.

        Examples:

        If the user wants to "enter" a right-side lab:
        Use the Right rule of that corridor.

        If the user wants to go to the "entrance" of a lab:
        Use the Forward direction of that corridor and stop at the door.

        --------------------------------------------------

        User Command:
        "{user_command}"

        --------------------------------------------------

        Output Rules:

        - Format: "ID,Angle"

        Output Example 1 (Navigation):
        door_1,180
        好的，我現在帶你去實驗室 214。

        Output Example 2 (Navigation):
        - **Multi-Goal Format**: If the user command involves multiple destinations or a sequence of tasks, 
        separate each "ID,Angle" with a semicolon (;).
        Example: "door_1,180;door_2,90"
        好的，那我現在先帶你去實驗室 214，在帶你去實驗室 213。

        Output Example 3 (Chatting):
        你好！我是實驗室助理機器人，很高興為你服務。

        - If no direction is specified:
        use the default angle of that corridor.

        - Return "None" if no matching instance is found.
        """
        try:
            # 準備當前的對話上下文
            # 將 system_context 放在第一筆對話或作為 instruction
            contents = []
            for entry in self.chat_history:
                contents.append(entry)
            
            # 加入使用者最新的一句話
            user_msg = {"role": "user", "parts": [{"text": user_command}]}
            contents.append(user_msg)

            # Gemini 推理 (帶入歷史紀錄)
            response = client.models.generate_content(
                model=CURRENT_MODEL,
                contents=contents,
                config={'system_instruction': system_context}
            )
            
            full_response = response.text.strip()
            self.get_logger().info(f'[### AI Response] AI 回應內容:\n{full_response}')

            # --- 解析回應內容 ---
            lines = full_response.split('\n')
            nav_data = ""
            spoken_text = ""

            for line in lines:
                if "," in line and any(char.isdigit() for char in line):
                    nav_data = line.strip()
                elif line.strip():
                    spoken_text += line.strip()

            # 1. 發布導航指令 (如果有)
            if nav_data:
                result_msg = String()
                result_msg.data = nav_data
                self.publisher_.publish(result_msg)
                self.get_logger().info(f'[### Navigation Pose] 發布導航目標: {nav_data}')

            # 2. 發布語音回覆給 TTS
            if spoken_text:
                tts_msg = String()
                tts_msg.data = spoken_text
                self.tts_publisher.publish(tts_msg)
                self.get_logger().info(f'[### Speech Responses] 發布語音回饋: {spoken_text}')

            # 3. 更新對話記憶 (只保留最近 10 次對話，避免 Token 過長)
            self.chat_history.append(user_msg)
            self.chat_history.append({"role": "model", "parts": [{"text": full_response}]})
            if len(self.chat_history) > 20: 
                self.chat_history = self.chat_history[-20:]

        except Exception as e:
            self.get_logger().error(f'⚠️ 推理異常: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LLMReasonerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()