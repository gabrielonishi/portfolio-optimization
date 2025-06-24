# Optimizing Portfolios using FP + Parallelization  

This projects runs a parallelized Monte Carlo simulation to find out what was the best performing class of 25 out of the 30 assets on the Dow Jones Industrial Average (DJIA). It uses Functional Programming concepts like pure functions in order to increase readibility, parallelization safety and debugging capabilities.

### Running the Project

This project uses Poetry to run. To install Poetry, run:
```bash
pip install pipx
pipx ensurepath
pipx install poetry
```

To install dependencies, use:
```bash
# On the /python dir
eval $(poetry env activate)
poetry install
```

Before running the project, take a look at the `config/settings.yaml` file. You can use it to setup number of processes and number of assets you wish to simulate, among other things.

To run the project, run;
```bash
python src/main.py
```

### Symmetric Encryption

As this is an academic project, the slides with the project description presented during class are not publicly available. Thus, they are protected with Symmetric Encryption using the GPG tool.

To decrypt:

``` 
gpg --decrypt docs/Aula17.pdf.gpg > docs/Aula17.pdf
```
