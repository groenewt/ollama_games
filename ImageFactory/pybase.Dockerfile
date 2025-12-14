FROM alpine/ollama AS llamasource
FROM rockylinux:9.3 AS base
#@TODO: ADD LABELS/ARGS
ENV PATH="/root/.local/bin:${PATH}"

#RUN dnf -y update
RUN dnf -y install \
        python3 python3-devel python3-pip \
        gcc gcc-c++ make \
        openssl-devel libffi-devel sqlite-devel \
        git nano mc
RUN dnf clean all

FROM base AS pyproject

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# @TODO: MAYBE? RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

COPY --from=llamasource /usr/lib/ollama/ /usr/lib/ollama/
COPY --from=llamasource /usr/bin/ollama /usr/bin/ollama



COPY ./config/ollama/model_list.txt /tmp/model_list.txt




WORKDIR /workspace

COPY ./config/ollama/ollama.env ollama.env



#@TODO: REMAINGING STEPS (Copy over from ./build -> 2. Scripts (./build/scripts/*->/scripts/* )

COPY ./scripts/init/pybase.init.sh /entrypoint.sh

RUN mkdir /projscripts
COPY ./scripts/utility /projscripts/utility
COPY ./scripts/init/service_init.sh /projscripts/init/service_init.sh

RUN chmod 777 /projscripts/init/service_init.sh
RUN chmod +x /projscripts/init/service_init.sh
RUN chmod +x /entrypoint.sh

#@TODO: MARIMO APP USER!

 #@TODO: REMAINGING STEPS (Copy over from ./build -> 3. workspace  (./build/workspace/*->/workspace/* )
COPY ./workspace .
RUN uv sync

EXPOSE 2718 11434 7241
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/sh"]