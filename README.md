#File kg.ipynb sẽ là file để load document, chunking, embedding và trích xuất thực thể, mối quan hệ để đưa lên neo4j

#Vì code trên google collab nên sẽ lưu vector database dưới dạng local nên mỗi lần chạy lại code phải thực hiện lại các bước từ đầu nên để đỡ tốn thời gian thì đã lưu dữ liệu ở 2 file một là documents_saved.pkl và entity_relaation_ok

#graph.html là hình biểu diễn đồ thị trên neo4j tuy nhiên vì các file toán chưa được xử lý đúng chuẩn nên đồ thị sẽ có sai sót

#file prompt.py để thực hiện các prompt yêu cầu llm xử lý

#file chatbot.py  để xử lý các tác vụ rag

#file app sẽ được demo trên streamlit. Tuy nhiên vẫn chưa ổn sẽ khác phụ hơn bằng cách viết api
