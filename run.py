from pml import app
app.run(debug=True)

from flask import Flask
app = Flask(__name__)

import pml.views
