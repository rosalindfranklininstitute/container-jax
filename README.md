# container-jax

Install as a module share module using `shpc`.

```
# Install
cd /path/to/registry
git clone git@github.com:rosalindfranklininstitute/container-jax.git
shpc install jax

# Update
cd /path/to/registry/jax
git pull
shpc install jax

# Usage
module load jax

# Run python within the container with jax available
python -c 'import jax'

# Run commands against the container
jax-run python -c 'import jax'

# Print the module usage help
module help jax
```

This module aliases `python` to the version within the container. 
Run commands or scripts prepended with `jax-run <command>` to access the container. 
