from flask import Flask, render_template, request, redirect, url_for
import subprocess  # Digunakan untuk menjalankan Streamlit dari Flask

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Halaman utama aplikasi web dengan pencarian ayat

@app.route('/search')
def search():
    # Panggil aplikasi Streamlit yang sudah Anda buat dengan "appendidikan_2.py"
    subprocess.run(["streamlit", "run", "appendidikan_2.py"])  # Pastikan pathnya benar
    return redirect(url_for('index'))  # Kembali ke halaman utama setelah pencarian

if __name__ == '__main__':
    app.run(debug=True)
