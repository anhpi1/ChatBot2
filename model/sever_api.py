from flask import Flask, request, jsonify
import du_doan as dd
from data.tham_so import tables
import suport as sp

app = Flask(__name__)

# Tải các mô hình
models = []
for name_mode in tables:
    new_model = sp.load_model(name_mode)
    models.append(new_model)

# API trả về dự đoán đầu
@app.route('/du_doan', methods=['POST'])
def du_doan():
    try:
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.get_json()
        
        # Kiểm tra xem có trường 'message' trong dữ liệu JSON không
        if 'message' not in data:
            return jsonify({"message": "No 'message' field found"}), 400
        
        # Lấy chuỗi ký tự
        message = data['message']
        
        # Trả về chuỗi ký tự nhận được
        return jsonify({"message": dd.final_du_doan(message, models)}), 200
    
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# API trả về đường dẫn report
@app.route('/lay_report', methods=['GET'])
def lay_report():
    dd.creater_report(models)
    return jsonify({"message_return": "model\\report"})


# API trả về dự đoán có lịch sử
@app.route('/du_doan_co_lich_su', methods=['POST'])
def du_doan_co_lich_su():
    try:
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.get_json()
        
        # Kiểm tra xem có trường 'message' và 'true_label' trong dữ liệu JSON không
        if 'message' not in data or 'true_label' not in data:
            return jsonify({"message": "No 'message' or 'true_label' field found"}), 400
        
        # Lấy chuỗi ký tự và nhãn đúng
        true_label = data['true_label']
        message = data['message']
        
        # Trả về chuỗi ký tự nhận được
        return jsonify({"message": dd.final_du_doan(message, models, true_label)}), 200
    
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
