from flask import Flask, render_template, request
import sympy as sp
import pandas as pd

app = Flask(__name__, static_folder='assets')

#Newton Raphson
def newton_raphson(f, df, x0, x1, tol, max_iter):
    result = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            result.append("Error: Turunan nol, metode gagal.")
            return result
        x1 = x1 - fx / dfx 
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
            func_input = request.form['func']
            x0 = float(request.form['x0'])
            x1 = float(request.form['x1'])
            tol = float(request.form['tol'])
            max_iter = int(request.form['max_iter'])
            
            x = sp.symbols('x')
            func = sp.sympify(func_input)
            dfunc = sp.diff(func, x)
            f = sp.lambdify(x, func)
            df = sp.lambdify(x, dfunc)
            
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

# Bisection
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

            result, root = metode_biseksi(f, a, b, tol, max_iter)
            if result is None:
                return render_template('bisection.html', result=[f"Error: {root}"])
            return render_template('bisection.html', result=result.to_dict(orient='records'), root=root)
        except Exception as e:
            return render_template('bisection.html', result=[f"Error: {e}"])
    return render_template('bisection.html', result=None)

# Regula Falsi
def regula_falsi(f, a, b, tol, max_iter):
    results = []
    if f(a) * f(b) > 0:
        return "Error: Fungsi tidak memiliki akar di dalam interval yang diberikan."

    for i in range(1, max_iter + 1):
        c = b - (f(b) * (b - a)) / (f(b) - f(a))
        error = abs(f(c))
        results.append([i, a, b, c, f(c), error])

        if error < tol:
            return results, f"Akar ditemukan: {c:.6f} dengan error: {error:.6f} setelah {i} iterasi."

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return results, "Metode tidak konvergen dalam batas iterasi maksimum."

@app.route('/regulasi', methods=['GET', 'POST'])
def regulasi():
    if request.method == 'POST':
        func_input = request.form['func']
        a = float(request.form['a'])
        b = float(request.form['b'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])

        x = sp.symbols('x')
        try:
            func = sp.sympify(func_input)
            f = sp.lambdify(x, func)
        except sp.SympifyError:
            return render_template('regulasi.html', message="Error: Fungsi tidak valid.")

        results, message = regula_falsi(f, a, b, tol, max_iter)
        
        return render_template('regulasi.html', results=results, message=message, func_input=func_input)

    return render_template('regulasi.html', results=None, message=None, func_input="")

# Secant
def secant_method(f, x0, x1, tol, max_iter):
    results = []
    for i in range(1, max_iter + 1):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if f_x1 - f_x0 == 0:
            results.append("Error: Pembagian dengan nol, metode gagal.")
            return results
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs(x2 - x1)
        results.append({
            'iterasi': i,
            'x0': x0,
            'x1': x1,
            'x2': x2,
            'f_x2': f(x2),
            'error': error
        })
        if error < tol:
            results.append(f"Akar ditemukan: {x2:.6f} dengan error: {error:.6f} setelah {i} iterasi.")
            return results
        x0, x1 = x1, x2
    results.append("Metode tidak konvergen dalam batas iterasi maksimum.")
    return results

@app.route('/secant', methods=['GET', 'POST'])
def secant():
    if request.method == 'POST':
        func_str = request.form['func']
        x0 = float(request.form['x0'])
        x1 = float(request.form['x1'])
        tol = float(request.form['tol'])
        max_iter = int(request.form['max_iter'])

        try:
            x = sp.symbols('x')
            func = sp.sympify(func_str)
            f = sp.lambdify(x, func, 'numpy')

            results = secant_method(f, x0, x1, tol, max_iter)
            return render_template('secant.html', results=results, error=None)

        except Exception as e:
            return render_template('secant.html', error=str(e), results=None)

    return render_template('secant.html', results=None, error=None)



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
