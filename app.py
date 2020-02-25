from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("hello.html")

@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'GET':
        data = request.files.get('myfile1','')

        print("only data", data)
        data.save("data", data.filename)
        print("Image saved")

        print(data.shape)
    # data = request.form['myfile1']
    return "Hello"


if __name__ == '__main__':
    app.run(debug=True)
