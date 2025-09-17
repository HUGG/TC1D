import nox


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    # install
    session.install(".[tests]")

    # See what is installed
    import pip  # needed to use the pip functions

    for i in pip.get_installed_distributions(local_only=True):
        print(i)

    # Run tests
    session.run("pytest")
