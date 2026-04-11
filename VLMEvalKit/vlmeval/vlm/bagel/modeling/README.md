# Modeling (VLMEvalKit Copy)

This `modeling/` directory is a **copy** of the top-level `CLEAR/modeling/` directory, vendored here so that VLMEvalKit evaluation scripts can import the BAGEL model architecture without modifying `sys.path` or requiring a package install.

## Keeping in sync

If you modify the model architecture in the top-level `modeling/` directory, you must copy the changes here as well:

```bash
# From the CLEAR project root:
rsync -av --delete modeling/ VLMEvalKit/vlmeval/vlm/bagel/modeling/
# Then restore this README:
git checkout -- VLMEvalKit/vlmeval/vlm/bagel/modeling/README.md
```
