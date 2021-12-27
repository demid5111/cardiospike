# 1. Update the system
sudo apt-get --yes update

# 2. Install dependencies for pyenv.
# We will need pyenv for installing particular Python version
sudo apt-get install --yes git make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
      libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# 3. Install pyenv
curl https://pyenv.run | bash

# 4. Activate pyenv
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 5. Install proper version of Python
export PYTHON_VERSION="3.9.1"
export is_installed=$(pyenv versions | grep $PYTHON_VERSION)

if [ -z "$is_installed" ]; then
  pyenv install $PYTHON_VERSION
fi

# 6. Use proper version of Python
pyenv global $PYTHON_VERSION

# 7. Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
poetry --version

# 8. Install project dependencies
#pyenv use $(PYTHON_VERSION)
poetry install
