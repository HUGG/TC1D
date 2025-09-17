import nox


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    # install
    session.install(".[tests]")

    # Run tests
    session.run("pytest")
