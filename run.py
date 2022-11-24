from pml import app
app.run(debug=True, port=5000)

from flask import Flask
app = Flask(__name__)

import pml.views
