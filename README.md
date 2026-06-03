# UCSD-CSE-234-Project-2

Team ID: 3

Team member: 

Yenan Xu PID: A69041466

Duan Wang PID: A69045640


### How to Run

```cli
python3 main.py --input input_filename --output output_filename
```

Additional flags: `--schemas_dir`, `--model_dir`, `--base_model`


### Model and adapter configuration path
We choose to use ***full checkpoint***.  

`adapter_config.json`, `adapter_model.satetensors`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json` are in the folder `checkpoint`. The `main.py` will call the adapter configuration and model parameters in this folder automatically. 





