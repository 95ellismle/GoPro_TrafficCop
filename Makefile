env: requirements.in
	python3.12 -m venv env
	. env/bin/activate
	pip install -r requirements.in

.PHONY: clean
clean:
	if [ -d env ]; then	rm -rf "env/"; fi
