# this is going to be a module in which default user interactions can be defined such that they cxan be easier, yet facily implemented in the main code.
from inputimeout import inputimeout, TimeoutOccurred
from datetime import datetime


def get_timestamp():

    now = datetime.now()

    return now.strftime("%Y-%m-%d %H:%M:%S\t")


# Websites to look up the individual codes:
# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
# https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
class Formatter():
    """
    A class for formatting console text with ANSI escape codes.

    Parameters
    ----------
    None

    Attributes
    ----------
    reset : str
        ANSI escape code for resetting text formatting.
    inv : str
        ANSI escape code for inverting text colors.
    rinv : str
        ANSI escape code for reverting inverted text colors.
    bold : str
        ANSI escape code for bold text.
    rbold : str
        ANSI escape code for reverting bold text.
    inf : str
        ANSI escape code for information messages.
    warning : str
        ANSI escape code for warning messages.
    error : str
        ANSI escape code for error messages.

    Methods
    -------
    warn(*messages)
        Print a warning message with appropriate formatting.
    err(*messages)
        Print an error message with appropriate formatting.
    info(*messages)
        Print an information message with appropriate formatting.

    Notes
    -----
    The class provides ANSI escape codes for common text formatting and colorization.
    It is designed for enhancing console output with colored messages.

    Examples
    --------
    >>> formatter = Formatter()
    >>> formatter.warn("This is a warning message.")
    >>> formatter.err("This is an error message.")
    >>> formatter.info("This is an information message.")
    """

    def __init__(self) -> None:
        """
        Initialize the Formatter class with ANSI escape codes for text formatting.
        """
        self.reset    = "\u001b[0m"
        self.inv      = "\u001b[7m"
        self.rinv     = "\u001b[27m"
        self.bold     = "\u001b[1m"
        self.rbold    = "\u001b[22m"
        self.blink    = "\u001b[5m"
        self.rblink    = "\u001b[25m"

        self.inf      = "\u001b[38;2;175;175;175m\u001b[1m"
        self.warning  = "\u001b[38;2;255;255;0m\u001b[1m"
        self.error    = "\u001b[38;2;255;50;50m\u001b[1m"


    def warn(self, *messages):
        """
        Print a warning message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.warning + "[WARNING]" + self.reset, *messages)
    

    def err(self, *messages):
        """
        Print an error message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.error + "[ERROR]" + self.reset, *messages)
    

    def info(self, *messages):
        """
        Print an information message with appropriate formatting.

        Parameters
        ----------
        *messages : tuple of str
            Variable number of message strings to be printed.
        """
        print(get_timestamp() + self.inf + "[INFO]" + self.reset, *messages)



def ask_user_with_timeout(
        question,
        positive_responses=["Y", "Yes", "YES", "y", "yes"],
        negative_responses=["N", "No", "NO", "y", "no"],
        timeout_seconds=None
    ):
    """
    Ask the user a yes/no question with specified positive and negative responses and optional timeout.

    Parameters
    ----------
    question : str
        The question to ask the user.
    positive_responses : list, optional
        List of valid positive responses (default is ["Y", "Yes", "YES", "y", "yes"]).
    negative_responses : list, optional
        List of valid negative responses (default is ["N", "No", "NO", "y", "no"]).
    timeout_seconds : int, float or None, optional
        Timeout duration in seconds. If None, there is no timeout (default is None).

    Returns
    -------
    bool
        True if the user's response is positive, False if it's negative or invalid.

    Raises
    ------
    TimeoutOccurred
        If the user does not respond within the specified timeout.

    """
    while True:
        try:
            user_input = inputimeout(prompt=f"{question} ({', '.join(positive_responses)}): ", timeout=timeout_seconds)
        except TimeoutOccurred:
            raise TimeoutOccurred("Timeout: User did not respond within the specified time.")

        user_input = user_input.strip().lower()

        if user_input in positive_responses:
            return True
        elif user_input in negative_responses:
            return False
        else:
            print("Invalid response. Please provide a valid response.")

if __name__ == "__main__":
    # Example usage:
    try:
        response = ask_user_with_timeout("Do you want to proceed?", ['y', 'yes'], ['n', 'no'], timeout_seconds=10)
        print(f"User's response: {response}")
    except TimeoutOccurred:
        print("Timeout: User did not respond within the specified time.")
