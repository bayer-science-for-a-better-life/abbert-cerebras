Vendored on 2022/02/12 from: https://github.com/Cerebras/modelzoo

Tag: R0.9.0_4

After this there was a big refactoring to accommodate frameworks other than TF in the namespaces.
We should ask Cerebras which version should we better use.

Changes:
  - Only vendored what we need to run our Antibody Bert models in inference.
  - Use namespace "abbert2.vendored.cerebras_modelzoo".
  - Accommodate TF versions newer than 2.2.0 (seemingly used by Cerebras, search for SANTI).
    The "Trainer" codepath is likely broken (changes are a bit moronic), but we should not hit it. 
