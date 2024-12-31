from flask import Flask, render_template, request
import sympy as sp
import pandas as pd

app = Flask(__name__)

def newton_raphson(f, df, x0, x1, tol, max_iter):
    result = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            result.append("Error: Turunan nol, metode gagal.")
            return result
        x1 = x1 - fx / dfx # x1 tetap seperti input
        error = abs(x1 - x0)
        result.append((i, x0, fx, dfx, x1, error))
        if error < tol:
            result.append(f"Akar ditemukan: {x1:.6f} dengan error: {error:.6f} setelah {i} iterasi.")
            return result
        x0 = x1
    result.append("Metode tidak konvergen dalam batas iterasi maksimum.")
    return result

@app.route('/newton', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Ambil input pengguna
            func_input = request.form['func']
            x0 = float(request.form['x0'])
            x1 = float(request.form['x1'])
            tol = float(request.form['tol'])
            max_iter = int(request.form['max_iter'])
            
            # Definisikan fungsi dari input pengguna
            x = sp.symbols('x')
            func = sp.sympify(func_input)
            dfunc = sp.diff(func, x)
            f = sp.lambdify(x, func)
            df = sp.lambdify(x, dfunc)
            
            # Jalankan metode Newton-Raphson
            result = newton_raphson(f, df, x0, x1, tol, max_iter)
            return render_template('newton.html', result=result)
        except Exception as e:
            return render_template('newton.html', result=[f"Error: {e}"])

    return render_template('newton.html', result=None)
def metode_biseksi(f, a, b, tol, max_iter):
    if f(a) * f(b) > 0:
        return None, "f(a) dan f(b) harus memiliki tanda berbeda."
    tabel_iterasi = []
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2
        tabel_iterasi.append([i + 1, a, b, c, fc, error])
        if error < tol or fc == 0:
            return pd.DataFrame(tabel_iterasi, columns=["Iterasi", "a", "b", "c", "f(c)", "Error"]), c
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return pd.DataFrame(tabel_iterasi, columns=["Iterasi", "a", "b", "c", "f(c)", "Error"]), None

@app.route('/bisection', methods=['GET', 'POST'])
def bisection():
    if request.method == 'POST':
        try:
            func_input = request.form['func']
            a = float(request.form['a'])
            b = float(request.form['b'])
            tol = float(request.form['tol'])
            max_iter = int(request.form['max_iter'])

            x = sp.symbols('x')
            func = sp.sympify(func_input)
            f = sp.lambdify(x, func)

            # Panggil metode biseksi
            result, root = metode_biseksi(f, a, b, tol, max_iter)
            if result is None:
                return render_template('bisection.html', result=[f"Error: {root}"])
            return render_template('bisection.html', result=result.to_dict(orient='records'), root=root)
        except Exception as e:
            return render_template('bisection.html', result=[f"Error: {e}"])
    return render_template('bisection.html', result=None)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
