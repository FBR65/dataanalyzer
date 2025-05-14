import io
import base64
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback
import logging
from decimal import Decimal, ROUND_HALF_UP  # Add ROUND_HALF_UP for proper rounding
import pandas as pd  # Add pandas import for DataFrame handling
import os
import datetime


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"--- Starting execution of {__name__} ---")
# --- End Logging Configuration ---

# --- Output Directory for Generated Code ---
GENERATED_CODE_DIR = "generated_code"
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)  # Ensure the directory exists
logger.info(f"Generated code will be saved to: {GENERATED_CODE_DIR}")
# --- End Output Directory ---


class PythonREPL:
    def __init__(self):
        self.exec_globals = {
            "pd": pd,  # Add pandas to globals
            "plt": plt,
            "np": None,
            "Decimal": Decimal,  # Make Decimal available
            "ROUND_HALF_UP": ROUND_HALF_UP,  # Add rounding constant
        }

    def run(self, code):
        logger.info(f"Executing Python code in REPL:\n```python\n{code}\n```")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            # Use the instance's exec_globals to maintain pandas and other imports
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
    """Execute Python code. Use matplotlib for visualization. Saves the code."""
    # Add imports for Decimal handling to the visualization code
    code = (
        """
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import matplotlib.pyplot as plt
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
            # Add a header comment for context (optional)
            f.write(f"# Generated code for data visualization at {timestamp}\n")
            f.write("# Tool: data_visualization\n\n")
            f.write(code)
        logger.info(f"Generated code saved to: {filename}")
    except Exception as e:
        # Log error but continue execution if possible
        logger.error(f"Failed to save generated code to {filename}: {e}")
    # --- End save code ---

    try:
        # Execute the code first to generate the plot
        logger.debug("Executing code for data visualization...")
        # Use the REPL to execute the code
        execution_output = repl.run(code)
        # Check if the execution itself returned an error message from the REPL
        if execution_output and execution_output.strip().startswith("Error:"):
            logger.error(f"Code execution failed within REPL: {execution_output}")
            # Return the execution error instead of trying to save a plot
            return f"Code execution failed:\n{execution_output}"

        logger.debug("Code execution finished. Saving plot to buffer...")

        # Check if a plot was actually created by the executed code
        if not plt.get_fignums():
            logger.warning("Code executed but no matplotlib figure was generated.")
            # Return the execution output and a warning, or just a warning message
            return f"Code executed successfully, but no plot was generated.\nExecution Output:\n{execution_output}"

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()  # Close the figure to free memory
        logger.info(
            "Data visualization created successfully (returning base64 string)."
        )
        # Optionally include execution output along with the image data URI if needed
        # return f"data:image/png;base64,{img_str}\nExecution Output:\n{execution_output}"
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        error_msg = f"Error creating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Data visualization failed. Error:\n{error_msg}")
        # Return the error message instead of the base64 string
        return error_msg


# Convert data to DataFrame
df = pd.DataFrame(
    [
        {"_id": "Berlin", "average_age": 30.0},
        {"_id": "Hamburg", "average_age": 30.0},
        {"_id": "Munich", "average_age": 28.0},
    ]
)

# Dynamically handle column names
if len(df.columns) >= 2:
    # Use the first column as the index and the second column as the y-axis
    index_column = df.columns[0]
    y_axis_column = df.columns[1]

    logger.info(
        f"Using '{index_column}' as the index and '{y_axis_column}' as the y-axis."
    )

    # Create appropriate visualization based on data
    plt.figure(figsize=(10, 6))

    # Set the index dynamically
    df.set_index(index_column, inplace=True)

    # Create bar plot
    df.plot(kind="bar", y=y_axis_column, legend=False)
    plt.xlabel(index_column.replace("_", " ").capitalize())
    plt.ylabel(y_axis_column.replace("_", " ").capitalize())
    plt.xticks(rotation=45)

    plt.title("Data Visualization")
    plt.tight_layout()
else:
    logger.warning(
        "Dataset does not have enough columns for visualization. Skipping plot creation."
    )
