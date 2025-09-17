import nox


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    # install
    session.install(".[tests]")

    # See what is installed
    import importlib.metadata

    distributions = importlib.metadata.distributions()
    installed_packages = []
    for dist in distributions:
        args = (dist.metadata["Name"], dist.version)
        installed_packages.append(args)
    installed_packages.sort()  # Sort the packages by name
    for package_name, version in installed_packages:
        print(f"{package_name}=={version}")

    # Run tests
    session.run("pytest")
