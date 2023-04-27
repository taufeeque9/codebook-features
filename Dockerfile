FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_UPSTREAM
ENV GIT_UPSTREAM=${GIT_UPSTREAM}
ARG GIT_BRANCH
ENV GIT_BRANCH=${GIT_BRANCH}
# Install OS dependencies
RUN apt update && apt install -y --no-install-recommends sudo git ssh openssh-server strace vim nano wget tmux curl nodejs && rm -rf /var/lib/apt/lists/*
# Initialize build-time git repositories (set $GIT_UPSTREAM and optionally $GIT_BRANCH)
COPY git_pull.sh /usr/local/bin/git_pull.sh
WORKDIR /root
# syntax = docker/dockerfile:1.3
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh /usr/local/bin/git_pull.sh
# Add the environment variables to the bashrc
RUN echo "export GIT_UPSTREAM="${GIT_UPSTREAM}"" >>.bashrc
RUN echo "export GIT_BRANCH="${GIT_BRANCH}"" >> .bashrc
# Enable auto-pull of git repo, potentially useful to build
RUN echo "/usr/local/bin/git_pull.sh" | tee -a .bashrc
# Provide a port by default, because it's useful
RUN pip install jupyter dill
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
#ENTRYPOINT ["/usr/bin/tini", "--"]

# add ssh key
COPY authorized_keys /root/.ssh/authorized_keys
RUN mkdir /run/sshd
RUN /usr/sbin/sshd

# create a user with the right UID etc.
ARG UID=1036
ARG USERNAME=mtaufeeque
RUN groupadd -g "$UID" "$USERNAME"
RUN useradd -r -d /homedir -s /bin/bash -u "$UID" -g "$USERNAME" "$USERNAME"
RUN mkdir -p /homedir && chown -R "$USERNAME:$USERNAME" /homedir
RUN usermod -aG sudo "$USERNAME"
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME

WORKDIR /data/codebook-features
COPY . /data/codebook-features
# RUN rm -rf outputs/ codebook_features/outputs
COPY trainer_pt_utils.py /opt/conda/lib/python3.10/site-packages/transformers/trainer_pt_utils.py
RUN git config --global --add safe.directory /data/codebook-features
RUN pip install -e .
RUN pip install pytest plotly 
RUN pip install git+https://github.com/neelnanda-io/PySvelte.git
RUN pip install git+https://github.com/taufeeque9/TransformerLens
WORKDIR /data/codebook-features/codebook_features

