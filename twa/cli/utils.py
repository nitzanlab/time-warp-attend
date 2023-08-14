import os
import glob
import click
import ruamel.yaml as yaml
from twa.utils import read_yaml
from typing import Type
import ast

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)
        
# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name: str) -> Type[click.Command]:
    """ Create and return a class inheriting `click.Command` which accepts a configuration file
        containing arguments/options accepted by the command.

        The returned class should be passed to the `@click.Commnad` parameter `cls`:

        ```
        @cli.command(name='command-name', cls=command_with_config('config_file'))
        ```

    Parameters:
        config_file_param_name (str): name of the parameter that accepts a configuration file

    Returns:
        class (Type[click.Command]): Class to use when constructing a new click.Command
    """

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params
                              if isinstance(p, click.core.Option)}
            param_defaults = {k: tuple(v) if type(v) is list else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if type(v) is list else v for k, v in ctx.params.items()}

            if config_file is not None:

                config_data = read_yaml(config_file)
                # modified to only use keys that are actually defined in options
                config_data = {k: tuple(v) if isinstance(v, yaml.comments.CommentedSeq) else v
                               for k, v in config_data.items() if k in param_defaults.keys()}

                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined

            return super().invoke(ctx)

    return custom_command_class


def get_command_defaults(command: click.Command):
    """ Get the default values for the options of `command`
    """
    return {tmp.name: tmp.default for tmp in command.params if not tmp.required}


def get_last_config(fname, suffix=''):
    """

    """
    if os.path.isfile(fname):
        return fname
    if os.path.isdir(fname):
        list_of_files = glob.glob(os.path.join(fname, '*%s.yaml' % suffix))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file





