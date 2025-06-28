import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns  # Add seaborn import
import sys
from io import StringIO
import traceback
import logging
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import os
import datetime


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"--- Starting execution of {__name__} ---")

GENERATED_CODE_DIR = "generated_code"
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
logger.info(f"Generated code will be saved to: {GENERATED_CODE_DIR}")


class PythonREPL:
    def __init__(self):
        self.exec_globals = {
            "pd": pd,
            "plt": plt,
            "sns": sns,  # Add seaborn to globals
            "np": None,
            "Decimal": Decimal,
            "ROUND_HALF_UP": ROUND_HALF_UP,
        }

    def run(self, code):
        logger.info(f"Executing Python code in REPL:\n```python\n{code}\n```")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            exec(code, self.exec_globals)
            sys.stdout = old_stdout
            result = redirected_output.getvalue()
            logger.info(f"REPL execution successful. Output:\n{result}")
            return result
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"REPL execution failed. Error:\n{error_msg}")
            return error_msg


async def data_visualization(code: str, repl: PythonREPL) -> str:
    """
    Execute Python code for data visualization.
    Args:
        code: str - The Python code to execute (including data handling)
        repl: PythonREPL - The REPL instance
    Returns:
        str - Base64 encoded image or error message
    """

    code = (
        """
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

"""
        + code
    )

    logger.info(f"Tool 'data_visualization' called with code:\n```python\n{code}\n```")

    # --- Save the generated code ---
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(
            GENERATED_CODE_DIR, f"visualization_code_{timestamp}.py"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Generated code for data visualization at {timestamp}\n")
            f.write("# Tool: data_visualization\n\n")
            f.write(code)
        logger.info(f"Generated code saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save generated code to {filename}: {e}")
    # --- End save code ---

    try:
        logger.debug("Executing code for data visualization...")

        execution_output = repl.run(code)

        if execution_output and execution_output.strip().startswith("Error:"):
            logger.error(f"Code execution failed within REPL: {execution_output}")

            return f"Code execution failed:\n{execution_output}"

        logger.debug("Code execution finished. Saving plot to buffer...")

        if not plt.get_fignums():
            logger.warning("Code executed but no matplotlib figure was generated.")

            return f"Code executed successfully, but no plot was generated.\nExecution Output:\n{execution_output}"

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        logger.info(
            "Data visualization created successfully (returning base64 string)."
        )
        # Optionally include execution output along with the image data URI if needed
        # return f"data:image/png;base64,{img_str}\nExecution Output:\n{execution_output}"
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        error_msg = f"Error creating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Data visualization failed. Error:\n{error_msg}")

        return error_msg
