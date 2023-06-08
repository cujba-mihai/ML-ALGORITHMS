# Activate the virtual environment
venv:
	. .venv/bin/activate

# Run the Flask application
run: venv
	flask run

.PHONY: venv run
