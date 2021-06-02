FROM python:3.7

WORKDIR /usr/src/app

RUN export HTTP_PROXY="http://192.168.49.1:8282" \
    && export http_proxy="http://192.168.49.1:8282" \
    https_proxy="http://192.168.49.1:8282" \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && echo "Acquire { HTTP::proxy \"http://192.168.49.1:8282\"; HTTPS::proxy \"http://192.168.49.1:8282\"; }" > /etc/apt/apt.conf \
    && apt install --no-install-recommends -y graphviz \
    && unset HTTP_PROXY http_proxy https_proxy

COPY ./.devcontainer/requirements.txt ./
# disable proxy options if you arent using it
RUN pip3 install --proxy=http://192.168.49.1:8282 --no-cache-dir -r ./requirements.txt

COPY ./code ./

VOLUME ["/static"]

EXPOSE 5000

CMD ["python", "./main.py"]