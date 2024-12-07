docker build -t python-311-env .
docker run -it --rm -v $(pwd):/app python-311-env
