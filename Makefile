env: requirements.in
	python3.12 -m venv env
	./env/bin/pip install -r requirements.in
	./env/bin/pip uninstall ffmpeg-python -y
	./env/bin/pip uninstall python-ffmpeg -y
	./env/bin/pip install ffmpeg-python --no-cache-dir --force-reinstall

.PHONY: clean
clean:
	if [ -d env ]; then	rm -rf "env/"; fi
