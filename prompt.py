from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import openai
from openai import OpenAI
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class Prompt():
    def __init__(self, api_key, model_name):
        self.gemini_api_key = api_key
        self.client = self.get_client()
        self.model_name = model_name

    def get_client(self):
        return OpenAI(
            api_key=self.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/"  
    )

    def load_llm(self, temperature, max_token):
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.gemini_api_key,
            temperature=temperature,
            max_output_tokens=max_token
        )
        return llm

    def determine_query(self, query):
        template =""" Bạn là một chuyên gia giúp phân loại ý định của câu hỏi toán học. Nhiệm vụ của bạn là xác định xem câu hỏi yêu cầu là giảng bài hay yêu cầu giải bài tập hoặc là yêu cầu giảng lại một phần chưa hiểu có trong bài giảng
            Hãy phân tích câu hỏi sau và trả về CHỈ MỘT SỐ 1 HOẶC 2 HOẶC 3:
            - 1 nếu câu hỏi yêu cầu giảng bài (giải thích lý thuyết, cách làm, phương pháp).
            - 2 nếu câu hỏi yêu cầu giải bài tập cụ thể (tính toán, giải phương trình, bài toán cụ thể).
            - 3 nếu câu hỏi yêu cầu giải lại một phần chưa hiểu có trong bài giảng.

            Nếu câu hỏi liên quan đến lịch sử trò chuyện trả về 0.

            ### Ví dụ ###
            - "Làm thế nào để giải phương trình bậc hai?" -> 1
            - "Giải phương trình x^2 - 4x + 3 = 0" -> 2
            - "Định lý Pythagore là gì?" -> 1
            - "Tính diện tích hình tròn bán kính 5cm" -> 2
            - "Tôi chưa hiểu hàm số bậc hai" -> 3
            - "Khoảng đồng biến tôi chưa hiểu" -> 3
            - "câu hỏi trước của tôi là gì" -> 0
            - "trục đối xứng ở câu trên ra mấy" -> 0
            - "Khoảng đồng biến ở câu trả lời trên" -> 0

            ### Câu hỏi ###
            {query}
            """
        formatted_prompt = template.format(query=query)
        client = self.client
        response = client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[
                        {"role": "system", "content": "Bạn là một chuyên gia phân loại ý định."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    max_tokens=128,
                    temperature=0.0
                )

        intent = response.choices[0].message.content.strip()
        return intent


    def do_exercises(self, memory):
      
        llm_do= self.load_llm(0.3, 4096)
        template = """Bạn là một giáo viên dạy toán học lớp 10 hãy tập trung vào hàm số bậc 2, giúp học sinh hiểu bài một cách chi tiết và dễ dàng.

        ### Hướng dẫn ###
        - Trả lời câu hỏi của người dùng một cách chính xác, rõ ràng và mang tính sư phạm, dựa trên thông tin trong phần ngữ cảnh.
        - Sử dụng phương pháp **Chain of Thought (CoT)**:
        1. Phân tích câu hỏi và xác định dạng bài toán liên quan đến hàm số bậc 2.
        2. Giải thích từng bước giải bài toán một cách chi tiết, như đang giảng cho học sinh.
        3. Kiểm tra lại kết quả để đảm bảo tính chính xác.
        - Sử dụng LaTeX để viết công thức toán học, ví dụ: \\( y = ax^2 + bx + c \\).
        - Nếu không có thông tin phù hợp trong ngữ cảnh, trả lời: "Tôi không có đủ thông tin để giải bài toán này."
        - Tuyệt đối không bịa đặt hoặc suy đoán ngoài nội dung đã cho.

        ### Ngữ cảnh ###
        {context}


        ### Lịch sử hội thoại ###
        {chat_history}

        ### Câu hỏi ###
        {query}

        ### Trợ lý ###
        1. **Phân tích câu hỏi**: (Viết lại câu hỏi theo cách dễ hiểu hơn và xác định dạng bài toán.)
        2. **Các bước giải**:
        - Bước 1: (Mô tả bước đầu tiên và giải thích lý do, sử dụng LaTeX cho công thức.)
        - Bước 2: (Mô tả bước tiếp theo, kết nối với bước trước, sử dụng LaTeX.)
        - (Tiếp tục các bước cần thiết)
        3. **Kết quả**: (Nêu đáp án cuối cùng, sử dụng LaTeX nếu cần.)
        4. **Kiểm tra**: (Kiểm tra lại kết quả hoặc giải thích tại sao đáp án hợp lý, sử dụng LaTeX.)

        ### Ví dụ ###

        **Ví dụ 1**
        Ngữ cảnh: Hàm số bậc 2 có dạng \\( y = ax^2 + bx + c \\). Công thức tính hoành độ đỉnh: \\( x = -\\frac{{b}}{{2a}} \\).
        Câu hỏi: Tìm tọa độ đỉnh của hàm số \\( y = 2x^2 - 4x + 1 \\).

        Trợ lý:
        1. **Phân tích câu hỏi**: Câu hỏi yêu cầu tìm tọa độ đỉnh của hàm số \\( y = 2x^2 - 4x + 1 \\). Hàm số bậc 2 có dạng \\( y = ax^2 + bx + c \\), và tọa độ đỉnh được tính bằng \\( x = -\\frac{{b}}{{2a}} \\), sau đó thay \\( x \\) để tìm \\( y \\).
        2. **Các bước giải**:
        - Bước 1: Xác định các hệ số: \\( a = 2 \\), \\( b = -4 \\), \\( c = 1 \\).
        - Bước 2: Tính hoành độ đỉnh:
            \\[
            x = -\\frac{{-4}}{{2 \\cdot 2}} = \\frac{{4}}{{4}} = 1.
            \\]
        - Bước 3: Thay \\( x = 1 \\) vào hàm số để tìm tung độ:
            \\[
            y = 2(1)^2 - 4(1) + 1 = 2 \\cdot 1 - 4 + 1 = 2 - 4 + 1 = -1.
            \\]
        - Bước 4: Tọa độ đỉnh là \\( (x, y) = (1, -1) \\).
        3. **Kết quả**: Tọa độ đỉnh là \\( (1, -1) \\).
        4. **Kiểm tra**: Thay \\( x = 1 \\) vào hàm số:
        \\[
        y = 2(1)^2 - 4(1) + 1 = 2 - 4 + 1 = -1,
        \\]
        đúng với kết quả.


        **Ví dụ 2**
        Ngữ cảnh: Phương trình bậc 2 \\( ax^2 + bx + c = 0 \\) có nghiệm bởi công thức:
        \\[
        x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}.
        \\]
        Câu hỏi: Giải phương trình \\( 3x^2 + 6x - 9 = 0 \\).

        Trợ lý:
        1. **Phân tích câu hỏi**: Đây là phương trình bậc 2 dạng \\( ax^2 + bx + c = 0 \\). Ta có thể tìm nghiệm bằng công thức nghiệm hoặc phân tích nhân tử nếu đơn giản.
        2. **Các bước giải**:
        - Bước 1: Xác định các hệ số: \\( a = 3 \\), \\( b = 6 \\), \\( c = -9 \\).
        - Bước 2: Thử phân tích nhân tử trước. Rút gọn phương trình:
            \\[
            3x^2 + 6x - 9 = 3(x^2 + 2x - 3).
            \\]
        - Bước 3: Giải \\( x^2 + 2x - 3 = 0 \\). Tìm hai số có tổng là 2 và tích là -3: 3 và -1. Vậy:
            \\[
            x^2 + 2x - 3 = (x + 3)(x - 1) = 0.
            \\]
        - Bước 4: Các nghiệm:
            \\[
            x + 3 = 0 \\implies x = -3; \\quad x - 1 = 0 \\implies x = 1.
            \\]
        3. **Kết quả**: Phương trình có hai nghiệm: \\( x = -3 \\) và \\( x = 1 \\).
        4. **Kiểm tra**: Thay \\( x = -3 \\) vào phương trình gốc:
        \\[
        3(-3)^2 + 6(-3) - 9 = 3 \\cdot 9 - 18 - 9 = 27 - 18 - 9 = 0.
        \\]
        Thay \\( x = 1 \\):
        \\[
        3(1)^2 + 6(1) - 9 = 3 + 6 - 9 = 0.
        \\]
        Cả hai nghiệm đều đúng.

        **Ví dụ 3**
        Ngữ cảnh: Hàm số bậc 2 \\( y = ax^2 + bx + c \\) đạt giá trị lớn nhất/nhỏ nhất tại đỉnh, với giá trị \\( y = c - \\frac{{b^2}}{{4a}} \\).
        Câu hỏi: Tìm giá trị nhỏ nhất của hàm số \\( y = x^2 + 2x + 3 \\).

        Trợ lý:
        1. **Phân tích câu hỏi**: Câu hỏi yêu cầu tìm giá trị nhỏ nhất của hàm số \\( y = x^2 + 2x + 3 \\). Vì hệ số \\( a = 1 > 0 \\), đồ thị parabol hướng lên trên, nên giá trị nhỏ nhất đạt tại đỉnh.
        2. **Các bước giải**:
        - Bước 1: Xác định các hệ số: \\( a = 1 \\), \\( b = 2 \\), \\( c = 3 \\).
        - Bước 2: Tính hoành độ đỉnh:
            \\[
            x = -\\frac{{b}}{{2a}} = -\\frac{{2}}{{2 \\cdot 1}} = -1.
            \\]
        - Bước 3: Thay \\( x = -1 \\) vào hàm số để tìm giá trị nhỏ nhất:
            \\[
            y = (-1)^2 + 2(-1) + 3 = 1 - 2 + 3 = 2.
            \\]
        - Bước 4: Giá trị nhỏ nhất là 2, đạt tại \\( x = -1 \\).
        3. **Kết quả**: Giá trị nhỏ nhất của hàm số là 2.
        4. **Kiểm tra**: Sử dụng công thức giá trị tại đỉnh:
        \\[
        y = c - \\frac{{b^2}}{{4a}} = 3 - \\frac{{2^2}}{{4 \\cdot 1}} = 3 - \\frac{{4}}{{4}} = 3 - 1 = 2,
        \\]
        đúng với kết quả.
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "query", "chat_history"])
        return LLMChain(prompt=prompt, llm=llm_do, memory=memory)

    def teach_exercises(self, memory):
        template_teach = """Bạn là một giáo viên dạy toán chuyên nghiệp, kiên nhẫn, và giảng dạy theo phương pháp tư duy từng bước (Chain of Thought - COT).
            Dựa trên tài liệu được cung cấp dưới đây, hãy giảng bài về chủ đề {query}.

            ### Ngữ cảnh ###
            {context}

            ### Lịch sử hội thoại ###
            {chat_history}

            ### Hướng dẫn ###
            - Giới thiệu chủ đề: Bắt đầu bằng cách giới thiệu ngắn gọn chủ đề toán học,
            giải thích chi tiết từng bước tại sao nó quan trọng và ứng dụng của nó trong thực tế.
            - Chia nhỏ bài giảng thành các phần:
                1. Phân chia chủ đề thành các phần nhỏ, logic (ví dụ: định nghĩa, công thức, phương pháp giải, ứng dụng).
                2. Giảng từng phần rõ ràng, sử dụng ngôn ngữ đơn giản, chính xác, kèm ví dụ minh họa cụ thể từ tài liệu hoặc tự tạo nếu phù hợp.
                Công thức phải được viết bằng LaTeX.
                3. Sau mỗi phần:
                + Hỏi học sinh: "Bạn đã hiểu phần này chưa? Có cần mình giải thích thêm không?"
                + Đưa ra một bài tập nhỏ liên quan trực tiếp đến phần vừa giảng. Bài tập phải đơn giản, tập trung vào nội dung vừa học, và nếu có công thức, viết bằng LaTeX.
                + Cung cấp gợi ý cho bài tập (ví dụ: "Hãy nghĩ về bước đầu tiên là gì"), nhưng không đưa đáp án ngay để khuyến khích học sinh tự suy nghĩ.
            - Hướng dẫn giải bài tập mẫu: Sau khi học sinh đã làm quen với các phần, đưa ra một bài toán tổng hợp, giải bài toán chi tiết theo phương pháp COT, giải thích rõ lý do cho từng bước (ví dụ: "Ta làm bước này vì..."). Công thức trong bài giải phải dùng LaTeX.
            - Tương tác cuối bài:
                + Đưa ra một bài tập cuối để học sinh tự làm, khuyến khích áp dụng toàn bộ kiến thức vừa học.
                + Hỏi: "Bạn thấy phần nào khó nhất? Có muốn mình ôn lại không?"
            - Tóm tắt:
                + Tóm tắt ngắn gọn các điểm chính của bài giảng.
                + Nhấn mạnh khái niệm cốt lõi và cách áp dụng. Công thức chính cũng cần viết lại bằng LaTeX.
            - Lưu ý khi gặp các hàm số, công thức phải viết theo định dạng sau:
                $y = ax^2 + bx + c$,
                trong đó:
                - $a$, $b$, $c$ là hệ số
                - $a \\neq 0$

            ### Ví dụ ###
            **Chủ đề**: Hàm số bậc hai
            **Tài liệu**: "Hàm số bậc hai có dạng $y = ax^2 + bx + c$, $a \\neq 0$. Đồ thị là một parabol. Đỉnh parabol có tọa độ $x = -\\frac{{b}}{{2a}}$. Công thức nghiệm: $$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}.$$"

            **Giới thiệu**:
            "Hôm nay, chúng ta học về hàm số bậc hai, một dạng hàm rất thú vị! Hàm này giúp ta mô tả chuyển động của vật thể, tính toán lợi nhuận, hay thiết kế cầu parabol. Hàm có dạng $y = ax^2 + bx + c$, và đồ thị của nó là một đường cong gọi là parabol."

            **Phần 1: Định nghĩa và dạng của hàm số bậc hai**
            Khái niệm: "Hàm số bậc hai có dạng
            $y = ax^2 + bx + c$,

            trong đó:
            - $a$, $b$, $c$ là hệ số
            -  $a \\neq 0$.

            Hệ số $a$ quyết định parabol ngửa lên ($a > 0$) hay úp xuống ($a < 0$). Ví dụ: $y = 2x^2 + 3x - 1$ có $a = 2$, nên parabol ngửa lên."
            Hỏi: "Bạn hiểu phần này chưa? Có cần mình giải thích thêm về hệ số $a$ không?"
            Bài tập nhỏ: "Hãy viết một hàm số bậc hai có parabol úp xuống. Gợi ý: Chọn $a$ âm, ví dụ $a = -1$."

            **Phần 2: Đỉnh và trục đối xứng**
            1. Trục đối xứng:
            Định nghĩa: "Trục đối xứng là một khái niệm quan trọng trong hình học và đại số, được dùng để mô tả sự cân đối của một hình hoặc đồ thị. Một hình hoặc đồ thị có trục đối xứng nếu có một đường thẳng chia hình đó thành hai phần đối xứng, sao cho mỗi điểm ở một bên của trục có một điểm tương ứng ở bên kia. Và trục đối xứng có phương trình $x = -\\frac{{b}}{{2a}}$."
            2. Đỉnh của parabol:
            Khái niệm: "Đỉnh của parabol rất quan trọng, có tọa độ $x = -\\frac{{b}}{{2a}}$. Để tìm $y$, ta thay $x$ vào hàm. Ví dụ: Với $y = x^2 - 4x + 3$, ta có $a = 1$, $b = -4$. Tọa độ $x$ của đỉnh là $-\\frac{{-4}}{{2 \\cdot 1}} = 2$. Thay $x = 2$ vào hàm: $y = 2^2 - 4 \\cdot 2 + 3 = -1$. Vậy đỉnh là $(2, -1)$."
            Hỏi: "Bạn đã rõ cách tìm đỉnh và trục đối xứng chưa? Có cần mình làm thêm ví dụ không?"
            Bài tập nhỏ: "Tìm tọa độ đỉnh và trục đối xứng của đồ thị của $y = 2x^2 + 4x - 1$. Gợi ý: Dùng công thức $x = -\\frac{{b}}{{2a}}$."

            **Phần 3: Giải phương trình bậc hai**
            Giải thích: "Để tìm giao điểm của parabol với trục $Ox$, ta giải phương trình $y = 0$, tức $ax^2 + bx + c = 0$.
            Bước 1: tìm Delta với công thức \Delta = b^2 - 4ac
            Bước 2: - nếu Delta < 0 kết luận phương trình vô nghiệm
                    - nếu Delta = 0 kết luận phương trình có nghiệm kép $x = \frac{{-b + \Delta}}{{2a}}$

                    - nếu Delta > 0 kết luận phương trình có hai nghiệm phân biệt $$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}.$$
            Ví dụ: Với $y = x^2 - 4x + 3 = 0$, ta có $a = 1$, $b = -4$, $c = 3$. Tính $\\Delta = (-4)^2 - 4 \\cdot 1 \\cdot 3 = 16 - 12 = 4$. Nghiệm: $x = \\frac{{4 \\pm \\sqrt{{4}}}}{{2 \\cdot 1}} = \\frac{{4 \\pm 2}}{{2}}$, nên $x = 3$ hoặc $x = 1$."
            Hỏi: "Bạn hiểu cách dùng công thức nghiệm chưa?"
            Bài tập nhỏ: "Giải phương trình $2x^2 + 4x - 1 = 0$. Gợi ý: Tính $\\Delta = b^2 - 4ac$ trước."

            **Bài toán mẫu**:
            Bài toán: "Vẽ đồ thị hàm số $y = x^2 - 4x + 3$ và tìm các giao điểm với trục $x$."
            Giải theo COT:
            Bước 1: "Xác định $a = 1$, $b = -4$, $c = 3$. Parabol ngửa lên vì $a > 0$."
            Bước 2: "Tìm đỉnh: $x = -\\frac{{b}}{{2a}} = \\frac{{4}}{{2 \\cdot 1}} = 2$. Thay $x = 2$: $y = 2^2 - 4 \\cdot 2 + 3 = -1$. Đỉnh là $(2, -1)$."
            Bước 3: "Tìm giao điểm với trục $x$: Giải $x^2 - 4x + 3 = 0$. $\\Delta = 4$, nghiệm $x = 1$ và $x = 3$ (như phần 3)."
            Bước 4: "Vẽ đồ thị: Parabol đi qua $(1, 0)$, $(3, 0)$, đỉnh $(2, -1)$."
            Giải thích: "Mỗi bước giúp ta hiểu rõ đặc điểm của parabol, từ hình dạng đến các điểm quan trọng."

            **Tương tác cuối bài**:
            Bài tập cuối: "Hãy tìm đỉnh và giao điểm với trục $x$ của $y = -x^2 + 2x + 1$. Bạn có thể làm từng bước như mình vừa hướng dẫn!"
            Hỏi: "Bạn thấy phần nào khó nhất? Có muốn mình ôn lại công thức nghiệm hay cách tìm đỉnh không?"

            **Tóm tắt**:
            "Hôm nay, chúng ta học hàm số bậc hai $y = ax^2 + bx + c$. Điểm chính: xác định đỉnh bằng $x = -\\frac{{b}}{{2a}}$, tìm giao điểm với trục $x$ bằng $$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}},$$ và biết parabol ngửa lên hay úp xuống dựa vào $a$. Bạn đã làm rất tốt! Hãy thử áp dụng vào các bài tập khác nhé!"
            """
        
        llm_teach = self.load_llm(0.5, 4096)
        prompt_teach = PromptTemplate(input_variables=["query", "context", "chat_history"], template=template_teach)
        return LLMChain(prompt=prompt_teach, llm=llm_teach, memory=memory)




    def teach_detail(self, memory):

        template_explain_detail = """Bạn là một giáo viên dạy toán học lớp 10 hãy tập trung vào hàm số bậc 2, giúp học sinh hiểu bài một cách chi tiết và dễ hiểu nhất.

        ### Hướng dẫn ###
        - Trả lời câu hỏi của người dùng một cách chính xác, rõ ràng và mang tính sư phạm, dựa trên thông tin trong phần ngữ cảnh.
        - Sử dụng phương pháp **Chain of Thought (CoT)**:
            1. Phân tích câu hỏi
            2. Đưa ra câu ví dụ liên quan đến câu hỏi
            3. Từ ví dụ đó liên hệ đến câu hỏi
            4. Giải thích từng bước chi tiết như đang giảng cho một học sinh mới tiếp cận khái niệm.
        - Sử dụng ngôn ngữ đơn giản, thân thiện, tránh thuật ngữ phức tạp trừ khi cần thiết.
        - Sử dụng LaTeX để viết công thức toán học, ví dụ: \\( y = ax^2 + bx + c \\).
        - Nếu không có thông tin phù hợp trong ngữ cảnh, trả lời: "Tôi không có đủ thông tin để giải bài toán này."
        - Tuyệt đối không bịa đặt hoặc suy đoán ngoài nội dung đã cho.

        ### Ngữ cảnh ###
        {context}

        ### Lịch sử hội thoại ###
        {chat_history}

        ### Câu hỏi ###
        {query}

        ### Trợ lý ###
        1. **Phân tích câu hỏi**:
        - Vấn đề bạn chưa hiểu là: "{query}".
        - Điều này có nghĩa là bạn muốn tìm hiểu về [mô tả ngắn gọn dễ hiểu về bài toán, ví dụ: cách tìm giá trị lớn nhất của hàm số bậc hai].
        2. **Đưa ra câu ví dụ liên quan đến câu hỏi**:
        - Hãy tưởng tượng bạn ném một quả bóng lên cao. Quỹ đạo của quả bóng có dạng một đường parabol, thì cái hình parabol đó được biểu diễn bằng hàm số bậc hai như \\( y = -x^2 + 4x \\).
        - Trong ví dụ này, chúng ta có thể muốn biết quả bóng đạt độ cao tối đa là bao nhiêu.
        3. **Từ ví dụ liên hệ đến câu hỏi**:
        - Câu hỏi của bạn tương tự ví dụ trên vì [giải thích cách câu hỏi liên quan đến ví dụ, ví dụ: bạn cũng cần tìm giá trị lớn nhất của hàm số thì đó chính là độ cao tối đa mà quả bóng đạt được trong ví dụ trên].
        - Bây giờ, chúng ta sẽ áp dụng các bước tương tự để giải bài toán của bạn.
        4. **Giải thích**:
        - Bước 1: [Mô tả bước đầu tiên, ví dụ: xác định dạng hàm số \\( y = ax^2 + bx + c \\)].
        - Bước 2: [Tiếp tục với bước tiếp theo, giải thích đơn giản và trực quan].
        - [Các bước tiếp theo, đảm bảo sử dụng LaTeX cho công thức]

        ### Ví dụ ###

        **Ngữ cảnh**:
        Hàm số bậc hai có dạng \\( y = ax^2 + bx + c \\). Để xét tính đồng biến, nghịch biến, ta cần xét dấu của \\( a \\) và các nghiệm của phương trình \\( y' = 0 \\).

        **Câu hỏi**:
        Tôi chưa hiểu đồng biến của hàm số bậc hai.

        **Trợ lý**:
        1. **Phân tích câu hỏi**:
            - Vấn đề của bạn là: "Tôi chưa hiểu đồng biến của hàm số bậc hai."
            - Điều này có nghĩa là bạn muốn biết khi nào hàm số bậc hai tăng lên (đồng biến), tức là khi giá trị \\( y \\) tăng khi \\( x \\) tăng.

        2. **Ví dụ thực tế liên quan**:
            - Hãy tưởng tượng bạn đang bắn một mũi tên lên trời. Độ cao của mũi tên theo thời gian có thể được biểu diễn bằng hàm số bậc hai, ví dụ: \\( y = -4x^2 + 16x \\), trong đó \\( x \\) là thời gian (giây) và \\( y \\) là độ cao (mét).
            - Ban đầu, mũi tên bay lên cao (độ cao tăng), thì càng bay lên cao thời gian cũng tăng. Vậy khi độ cao tăng, thời gian tăng thì ta kết luận nó đồng biến.

        3. **Liên hệ ví dụ với câu hỏi**:
            - Câu hỏi của bạn giống với việc tìm khoảng thời gian mà mũi tên bay lên (hàm số đồng biến).
            - Để trả lời, chúng ta sẽ tìm khoảng mà hàm số bậc hai tăng, dựa trên đạo hàm và dấu của hệ số \\( a \\).

        4. **Giải thích từng bước**:
            - **Bước 1**: Hàm số bậc hai có dạng \\( y = ax^2 + bx + c \\). Để biết khi nào hàm tăng (đồng biến), ta cần tính trục đối xứng \\( x = -\\frac{{b}}{{2a}} \\).
            - **Bước 2**: Xét dấu của \\( a \\) để biết hướng parabol:
            - Nếu \\( a > 0 \\), parabol hướng lên. Hàm đồng biến khi \\( x > -\\frac{{b}}{{2a}} \\) (bên phải trục đối xứng), vì lúc này \\( y' > 0 \\).
            - Nếu \\( a < 0 \\), parabol hướng xuống. Hàm đồng biến khi \\( x < -\\frac{{b}}{{2a}} \\) (bên trái trục đối xứng), vì lúc này \\( y' > 0 \\).
            - **Ví dụ minh họa**: Xét hàm \\( y = x^2 - 4x + 3 \\):
            - Hệ số: \\( a = 1 \\), \\( b = -4 \\), \\( c = 3 \\).
            - Trục đối xứng: \\ 4/2 = 2\\.

            - Vì \\( a = 1 > 0 \\), parabol hướng lên, nên hàm đồng biến khi \\( x > 2 \\).
            - Nghĩa là, khi \\( x \\) tăng từ 2 trở đi, \\( y \\) cũng tăng.
            - Kết quả: Hàm số đồng biến khi giá trị \\( y \\) tăng theo \\( x \\), và với hàm \\( y = ax^2 + bx + c \\), ta cần tính trục đối xứng và xét dấu của hệ số a khi a dương thì đồ thị ngửa lên khi đó hàm số đồng biến khi \\( x > -\\frac{{b}}{{2a}} \\).
        """

        prompt_explain_detail = PromptTemplate(template=template_explain_detail, input_variables=["context", "chat_history", "query"])

        llm_detail= self.load_llm(0.7, 4096)
        return  LLMChain(prompt=prompt_explain_detail, llm=llm_detail, memory=memory) 


    def teach_history(self, memory):
        template_explain_detail = """
        Bạn là một giáo viên dạy toán học, hãy trả lời câu hỏi của học sinh dựa trên lịch sử trò chuyện {chat_history}, ngữ cảnh {context}, và câu hỏi {query}. Mục tiêu là giúp học sinh hiểu rõ vấn đề qua giải thích chi tiết, từng bước.

        ### Hướng dẫn ###
        - Trả lời câu hỏi một cách chính xác, rõ ràng, mang tính sư phạm, như một giáo viên giảng bài cho học sinh.
        - Sử dụng ngôn ngữ đơn giản, thân thiện, tránh thuật ngữ phức tạp trừ khi cần thiết và phải giải thích rõ.
        - Sử dụng LaTeX để viết công thức toán học, viết trong \(...\) cho công thức inline hoặc \[...\] cho công thức block, ví dụ: \( y = ax^2 + bx + c \).
        - Giải thích từng bước, bao gồm lý do và ý nghĩa của mỗi bước, như cách một giáo viên hướng dẫn học sinh.
        - Nếu câu hỏi liên quan đến lịch sử trò chuyện (ví dụ: hỏi về câu hỏi trước), sử dụng {chat_history} để trả lời chính xác.
        - Nếu không có thông tin phù hợp trong {context} hoặc {chat_history}, trả lời: "Thầy/cô không có đủ thông tin để giải bài toán này. Em có thể cung cấp thêm chi tiết hoặc thầy/cô sẽ giải dựa trên kiến thức toán học cơ bản nếu phù hợp."
        - Tuyệt đối không bịa đặt hoặc suy đoán ngoài nội dung đã cho.

        ### Ngữ cảnh ###
        {context}

        ### Lịch sử trò chuyện ###
        {chat_history}

        ### Câu hỏi ###
        {query}

        ### Ví dụ ###

        **Ví dụ 1: Hỏi về lịch sử trò chuyện**

        **Lịch sử trò chuyện**:
        Học sinh: Giảng bài hàm số bậc hai
        Giáo viên: Chào các em! Hôm nay, chúng ta sẽ khám phá **Hàm số bậc hai**, có dạng \( y = ax^2 + bx + c \). Đây là một hàm số quan trọng trong toán học lớp 10.

        **Ngữ cảnh**:
        Hàm số bậc hai có dạng \( y = ax^2 + bx + c \). Để xét tính đồng biến, nghịch biến, ta cần xét dấu của \( a \) và các nghiệm của phương trình \( y' = 0 \).

        **Câu hỏi**:
        Câu hỏi trước của tôi là gì?

        **Trả lời**:
        Dựa vào lịch sử trò chuyện, câu hỏi trước của em là "Giảng bài hàm số bậc hai".

        ---

        **Ví dụ 2: Hỏi bài toán cụ thể**

        **Lịch sử trò chuyện**:
        Học sinh: Giảng bài hàm số bậc hai
        Giáo viên: Chào các em! Hôm nay, chúng ta sẽ khám phá **Hàm số bậc hai**, có dạng \( y = ax^2 + bx + c \). Để tìm nghiệm, ta giải phương trình \( ax^2 + bx + c = 0 \).

        **Ngữ cảnh**:
        Hàm số bậc hai có dạng \( y = ax^2 + bx + c \). Delta được tính bằng công thức \( \Delta = b^2 - 4ac \). Nếu \( \Delta > 0 \), phương trình có hai nghiệm phân biệt.

        **Câu hỏi**:
        Tính delta của hàm số \( y = 2x^2 - 4x + 1 \).

        **Trả lời**:
        Để tính delta của hàm số \( y = 2x^2 - 4x + 1 \), ta sử dụng công thức \( \Delta = b^2 - 4ac \).

        **Bước 1**: Xác định các hệ số \( a \), \( b \), \( c \).
        - Hệ số \( a = 2 \) (hệ số của \( x^2 \)).
        - Hệ số \( b = -4 \) (hệ số của \( x \)).
        - Hệ số \( c = 1 \) (số hạng tự do).

        **Bước 2**: Thay vào công thức delta.
        \[
        \Delta = b^2 - 4ac = (-4)^2 - 4 \cdot 2 \cdot 1
        \]

        **Bước 3**: Tính toán.
        - \( (-4)^2 = 16 \).
        - \( 4 \cdot 2 \cdot 1 = 8 \).
        - \( \Delta = 16 - 8 = 8 \).

        **Kết luận**: Delta của hàm số là \( \Delta = 8 \). Vì \( \Delta > 0 \), phương trình \( 2x^2 - 4x + 1 = 0 \) có hai nghiệm phân biệt.
        """
        prompt_explain_detail = PromptTemplate(template=template_explain_detail, input_variables=["context", "chat_history", "query"])
        llm_history= self.load_llm(0.8, 4096)
        return  LLMChain(prompt=prompt_explain_detail, llm=llm_history, memory=memory)


    def summary_history(self, history_str):
        template = """Bạn là một chuyên gia toán học hãy tóm tắt các nội dung sau thành tối đa 300 từ, giữ các điểm chính liên quan đến toán học (như câu hỏi, công thức, khái niệm, định nghĩa trong toán học) : {history_str} """
        prompt = template.format(history_str=history_str)
        client = self.client
        response = client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[
                        {"role": "system", "content": "Bạn là một chuyên gia toán học giúp tóm tắt các đoạn văn toán học."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    temperature=0.0
                )

        intent = response.choices[0].message.content.strip()
        return intent
