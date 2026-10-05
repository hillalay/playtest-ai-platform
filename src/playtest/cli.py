import typer

app = typer.Typer()


@app.command()
def hello():
    print("Playtest AI is running")


if __name__ == "__main__":
    app()