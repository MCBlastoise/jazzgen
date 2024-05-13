if [ ! -d "jazzgen-env" ]; then
    python3 -m venv jazzgen-env
    echo "venv created."
else
    echo "venv already exists."
fi

source jazzgen-env/bin/activate

pip install -r requirements.txt