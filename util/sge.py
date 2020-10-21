import os
from shutil import copyfile, rmtree


def run_one_job(script,
                script_parameters=[('log_dir', 'test')],
                queue='middle',
                out_dir='/out/dir',
                gpu=1,
                hostname='*',
                cpu_only=False,
                array=False,
                num_jobs=10,
                name='', memory=50, overwrite=False, hold_off=False, num_cpu_cores=1):
    """Copies script to execute to new location and submits it via qsub

    Parameters:
    script (string): Full path to script which shall be executed
    script_parameters (list of pairs): Parameters passed to script at execution
    queue (string): CPU: short, middle, long GPU: 2h, 24h, 48h, 5d
    out_dir (string): Full path to directory where script and log are placed
    gpu (int): How many gpus to use
    hostname (regex): limit hosts to deploy on (default '*')
    cpu_only (boolean): only use cpu, not gpu (default False)
    name (string): name of job, will be set automatically if not provided
    memory (int): GB RAM for job (default 50)

    Returns:
   """

    # Make output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not overwrite:
        print('Experiment already exists:', out_dir, 'Will not re-run.')
        return
    else:
        rmtree(out_dir)
        os.makedirs(out_dir)

    if name == '':
        name = os.path.basename(os.path.dirname(out_dir)) + '_' + os.path.basename(out_dir)

    # Copy executable file (to allow changes to original while job is waiting in queue)
    script_to_execute = os.path.join(out_dir, os.path.basename(script))
    copyfile(script, script_to_execute)
    os.system("chmod u+x " + script_to_execute)

    # Create batch job file
    for par_name, par_value in script_parameters:
        script_to_execute = script_to_execute + ' --' + par_name + ' ' + str(par_value)
    with open(os.path.join(out_dir, 'qsub.sh'), 'w') as script:
        if not cpu_only:
            if not array:
                script.write(get_qusub_script(name, gpu, hostname, queue, out_dir, script_to_execute, memory))
            else:
                script.write(
                    get_qusub_array_script(name, gpu, hostname, queue, out_dir, script_to_execute, num_jobs, memory))
        else:
            if not array:
                script.write(get_cpu_qusub_script(name, hostname, queue, out_dir, script_to_execute, memory))
            else:
                script.write(get_cpu_array_script(name, hostname, queue, out_dir, script_to_execute, num_jobs, memory))

    os.system("chmod u+x " + os.path.join(out_dir, 'qsub.sh'))
    if not hold_off:
        os.system("qsub " + os.path.join(out_dir, 'qsub.sh'))


def get_qusub_script(name, gpu, hostname, queue, out_dir, script_to_execute, memory=80):
    s = "#!/bin/bash\n" \
        "#\n" \
        "#$ -N {}\n" \
        "#\n" \
        "## otherwise the default shell would be used\n" \
        "#$ -S /bin/bash\n" \
        "\n" \
        "## demand gpu resource\n" \
        "#$ -l gpu={}\n" \
        "#$ -l hostname={}\n" \
        "\n" \
        "## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue\n" \
        "#$ -q gpu.{}.q@*\n" \
        "\n" \
        "## the maximum memory usage of this job, (below 4G does not make much sense)\n" \
        "#$ -l h_vmem={}G\n" \
        "\n" \
        "## stderr and stdout are merged together to stdout\n" \
        "#$ -j y\n" \
        "\n" \
        "## logging directory. preferrably on your scratch\n" \
        "#$ -o {}\n" \
        "\n" \
        "# if you need to export custom libs, you can do that here\n" \
        "source /usr/Setpath.sh\n" \
        "\n" \
        "# call your calculation executable, redirect output\n" \
        "{}\n".format(name, gpu, hostname, queue, memory, out_dir, script_to_execute)
    return s


def get_qusub_array_script(name, gpu, hostname, queue, out_dir, script_to_execute, num_jobs, memory=80):
    s = "#!/bin/bash\n" \
        "#\n" \
        "#$ -N {}\n" \
        "#\n" \
        "## otherwise the default shell would be used\n" \
        "#$ -S /bin/bash\n" \
        "\n" \
        "## demand gpu resource\n" \
        "#$ -l gpu={}\n" \
        "#$ -l hostname={}\n" \
        "\n" \
        "## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue\n" \
        "#$ -q gpu.{}.q@*\n" \
        "\n" \
        "## the maximum memory usage of this job, (below 4G does not make much sense)\n" \
        "#$ -l h_vmem={}G\n" \
        "\n" \
        "## stderr and stdout are merged together to stdout\n" \
        "#$ -j y\n" \
        "\n" \
        "## logging directory. preferrably on your scratch\n" \
        "#$ -o {}\n" \
        "\n" \
        "## schedule 10 jobs with ids 1-{}\n" \
        "#$ -t 1-{}\n" \
        "\n" \
        "# if this job is run in sge: take the sge task id\n" \
        "# otherwise, the first argument of this script is taken as task id (if you want \n" \
        "# to try the script locally).\n" \
        "TASK_ID=${{SGE_TASK_ID:-\"$1\"}}\n" \
        "\n" \
        "# if you need to export custom libs, you can do that here\n" \
        "source /usr/Setpath.sh\n" \
        "\n" \
        "# call your calculation executable, redirect output\n" \
        "{} --task_id $TASK_ID\n".format(name, gpu, hostname, queue, memory, out_dir, num_jobs, num_jobs,
                                         script_to_execute)
    return s


def get_cpu_qusub_script(name, hostname, queue, out_dir, script_to_execute, memory=20):
    s = "#!/bin/bash\n" \
        "#\n" \
        "#$ -N {}\n" \
        "#\n" \
        "## otherwise the default shell would be used\n" \
        "#$ -S /bin/bash\n" \
        "\n" \
        "## demand gpu resource\n" \
        "#$ -l hostname={}\n" \
        "\n" \
        "## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue\n" \
        "#$ -q {}.q@*\n" \
        "\n" \
        "## the maximum memory usage of this job, (below 4G does not make much sense)\n" \
        "#$ -l h_vmem={}G\n" \
        "\n" \
        "## stderr and stdout are merged together to stdout\n" \
        "#$ -j y\n" \
        "\n" \
        "# logging directory. preferrably on your scratch\n" \
        "#$ -o {}\n" \
        "\n" \
        "# if you need to export custom libs, you can do that here\n" \
        "source /usr/Setpath.sh\n" \
        "\n" \
        "# call your calculation executable, redirect output\n" \
        "{}\n".format(name, hostname, queue, memory, out_dir, script_to_execute)
    return s


def get_cpu_array_script(name, hostname, queue, out_dir, script_to_execute, num_jobs, memory=20):
    s = "#!/bin/bash\n" \
        "#\n" \
        "#$ -N {}\n" \
        "#\n" \
        "## otherwise the default shell would be used\n" \
        "#$ -S /bin/bash\n" \
        "\n" \
        "## demand gpu resource\n" \
        "#$ -l hostname={}\n" \
        "\n" \
        "## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue\n" \
        "#$ -q {}.q@*\n" \
        "\n" \
        "## the maximum memory usage of this job, (below 4G does not make much sense)\n" \
        "#$ -l h_vmem={}G\n" \
        "\n" \
        "## stderr and stdout are merged together to stdout\n" \
        "#$ -j y\n" \
        "\n" \
        "# logging directory. preferably on your scratch\n" \
        "#$ -o {}\n" \
        "\n" \
        "## schedule 10 jobs with ids 1-{}\n" \
        "#$ -t 1-{}\n" \
        "\n" \
        "# if this job is run in sge: take the sge task id\n" \
        "# otherwise, the first argument of this script is taken as task id (if you want \n" \
        "# to try the script locally).\n" \
        "TASK_ID=${{SGE_TASK_ID:-\"$1\"}}\n" \
        "\n" \
        "# if you need to export custom libs, you can do that here\n" \
        "source /usr/Setpath.sh\n" \
        "\n" \
        "# call your calculation executable, redirect output\n" \
        "{} --task_id $TASK_ID\n".format(name, hostname, queue, memory, out_dir, num_jobs, num_jobs, script_to_execute)
    return s


if __name__ == "__main__":
    print(get_cpu_array_script('bla', 'bla', 'bla', 'bla', 'bla', 30))
    # run_one_job()
