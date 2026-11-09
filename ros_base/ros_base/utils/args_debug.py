import argparse
import sys

import debugpy


def setup_debugging(listen_port: int = 5678):
    ip_address = ("0.0.0.0", listen_port)
    print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
    print(f"Process: {sys.argv[:]}")
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

    return None


def add_debug_mode(args=None, listen_port: int = 5678):
    if args is not None:
        if not getattr(args, "debug", False):
            print(f"Process: {sys.argv[:]}")
        else:
            setup_debugging(listen_port=listen_port)
        return None

    else:
        if "--debug" in sys.argv:
            setup_debugging(listen_port=listen_port)
        # Create a new parser to add the --debug argument
        parser = argparse.ArgumentParser()
        # Return the parser to allow further argument additions
        parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

        return parser


if __name__ == "__main__":
    parser = add_debug_mode(listen_port=9999)
    parser.add_argument("--value", type=int, default=0, help="A test argument.")
    args = parser.parse_args()
    print(f"Debug mode is set up. value: {args.value}")
