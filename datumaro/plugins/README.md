# Add a new builtin plugin

If you are adding a new builtin plugin under `./datumaro/plugins/`, you have to enroll it to `./datumaro/plugins/specs.json` for the lazy import.
Otherwise, you will be failed in this test (`tests/unit/test_environment.py`).
You can enroll your plugin to `./datumaro/plugins/specs.json` with Python executable script as follows.

```console
python datumaro/plugins/specs.py
```

> **_NOTE:_**  It is not recommended to mannually modify `./datumaro/plugins/specs.json` by hands. Please use the above Python executable script.
