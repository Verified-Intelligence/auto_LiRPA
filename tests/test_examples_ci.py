import subprocess
import traceback

import test_examples

original_subprocess_run = subprocess.run


def custom_run(*args, **kwargs):
    kwargs.setdefault('check', True)
    return original_subprocess_run(*args, **kwargs)


subprocess.run = custom_run


def run_tests():
    # get all func start with test in test_examples other than 'test_release'
    # and 'test_cifar_training'(cannot run on GPU with memory lower than 32GB)
    test_functions = [
        getattr(test_examples, func) for func in dir(test_examples)
        if callable(getattr(test_examples, func)) and func.startswith('test')
        and func not in ['test_release']
    ]

    try:
        for test_func in test_functions:
            test_func()
            print(f"{test_func.__name__} executed successfully.")
    except Exception as e:
        print(f"Exception in {test_func.__name__}: {e}")
        traceback.print_exc()  # Print detailed exception information

        print("Examples Test Result:")
        print("\nFailed tests:")
        print(test_func.__name__)
        raise

    print("Examples Test Result:")
    print("\nAll tests passed successfully.")


if __name__ == '__main__':
    run_tests()
