Step 1: Install multiworld package
```
$ pip install .

```

Step 2: run post installation script with patch file.
```
m8d-post-setup <path_to_site_packages>
```

Patch files exist under `patch` folder.
Example:
```
m8d-post-setup patch/pytorch-v2.2.1.patch
```
The version (v2.2.1) must match the installed pytorch version.
