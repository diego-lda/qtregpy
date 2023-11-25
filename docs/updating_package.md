# Updating a Python Package Release on PyPI

This guide provides a step-by-step approach to updating the `qtregpy` Python package on the Python Package Index (PyPI). This process involves modifying the package version, building the package, and then uploading it to PyPI.

## Step 1: Update the setup.py File

Once all new changes have been made, and you are happy with them, you need to make sure they are all added to the remote repo. Before uploading the new version of the package, you need to update the version number in your setup.py file. This is essential for PyPI and your users to recognize it as a new release.

1. Open your setup.py file.

2. Locate the version argument in the setup() function. It usually looks like this:

    ```python
    setup(
        ...
        version='0.1.1',
        ...
    )
    ```

3. Also update the download_url to point to the new version tag. For example:

    ```python
    setup(
        ...
        download_url='https://github.com/diego-lda/qtregpy/archive/refs/tags/0.1.1.tar.gz',
        ...
    )
    ```

4. Increment the version number following semantic versioning principles. For example, change 0.1.0 to 0.1.1 for a minor update.

5. Save the changes to setup.py.

### Step 2: Commit Changes to GitHub

After updating the version number, commit your changes and push them to your GitHub repository.

1. Stage your changes:

    ```bash
    git add setup.py
    ```

2. Commit the changes with a meaningful message:

    ```bash
    git commit -m "Update version to 0.1.1"
    ```

3. Push the commit to GitHub:

    ```bash
    git push origin main
    ```

## Step 3: Tag the Release on GitHub

After pushing the changes, tag the release on GitHub.

1. Tag the release using the new version number:

    ```bash
    git tag 0.1.1
    git push origin 0.1.1
    ```

2. Then go to the tag on the GitHub website and create a new release.

Alternatively, you can create a new release via the GitHub web interface.

## Step 4: Build Your Package

After updating the version number, you need to build your package.

1. Ensure you have the required tools installed:

    ```bash
    pip install setuptools wheel twine
    ```

2. Run the following commands in your project directory to build your package:

    ```bash
    python setup.py sdist bdist_wheel
    ```

    This will create source and wheel distributions in the dist/ directory.

## Step 5: Upload the New Version to PyPI

Use twine to upload your package to PyPI. Ensure you have your PyPI credentials or API token ready.

1. If you're not using an API token, log in to PyPI and create one under the account settings.

2. Use the following command to upload your package:

    ```bash
    twine upload dist/*
    ```

3. When prompted, enter your PyPI username and password or use your API token (username: __token__, password: your API token).

4. Alternatively you can use the token directly:

    ```bash
    twine upload dist/* -u __token__ -p <your_token_here>
    ```

## Step 6: Verify the Update

1. After uploading, check your package on the PyPI website to ensure the new version is listed.

2. Optionally, install your package using pip to test:

    ```bash
    pip install qtregpy==New.Version.Number
    ```

## Conclusion

You've now successfully updated the `qtregpy` Python package on PyPI. Remember to update any documentation or release notes as necessary.
