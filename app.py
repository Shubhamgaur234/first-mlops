from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import LoanData, LoanDataClassifier
from src.pipeline.training_pipeline import TrainPipeline


load_dotenv()

app = FastAPI(title="Loan Prediction Pipeline")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _loan_data_from_mapping(data: dict) -> LoanData:
    return LoanData(
        ID=_to_float(data.get("ID")),
        year=_to_float(data.get("year")),
        loan_limit=data.get("loan_limit"),
        Gender=data.get("Gender"),
        approv_in_adv=data.get("approv_in_adv"),
        loan_type=data.get("loan_type"),
        loan_purpose=data.get("loan_purpose"),
        Credit_Worthiness=data.get("Credit_Worthiness"),
        open_credit=data.get("open_credit"),
        business_or_commercial=data.get("business_or_commercial"),
        loan_amount=_to_float(data.get("loan_amount")),
        rate_of_interest=_to_float(data.get("rate_of_interest")),
        Interest_rate_spread=_to_float(data.get("Interest_rate_spread")),
        Upfront_charges=_to_float(data.get("Upfront_charges")),
        term=_to_float(data.get("term")),
        Neg_ammortization=data.get("Neg_ammortization"),
        interest_only=data.get("interest_only"),
        lump_sum_payment=data.get("lump_sum_payment"),
        property_value=_to_float(data.get("property_value")),
        construction_type=data.get("construction_type"),
        occupancy_type=data.get("occupancy_type"),
        Secured_by=data.get("Secured_by"),
        total_units=data.get("total_units"),
        income=_to_float(data.get("income")),
        credit_type=data.get("credit_type"),
        Credit_Score=_to_float(data.get("Credit_Score")),
        co_applicant_credit_type=data.get("co-applicant_credit_type")
        or data.get("co_applicant_credit_type"),
        age=data.get("age"),
        submission_of_application=data.get("submission_of_application"),
        LTV=_to_float(data.get("LTV")),
        Region=data.get("Region"),
        Security_Type=data.get("Security_Type"),
        dtir1=_to_float(data.get("dtir1")),
    )


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "loanTemplate.html",
        {"context": ""},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/train")
async def train_route_client():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful")
    except Exception as e:
        return Response(f"Error occurred: {e}", status_code=500)


@app.post("/")
async def predict_route_client(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
        else:
            payload = dict(await request.form())

        loan_data = _loan_data_from_mapping(payload)
        loan_df = loan_data.get_loan_input_data_frame()
        prediction = LoanDataClassifier().predict(dataframe=loan_df)[0]
        status = f"Prediction: {prediction}"

        if "application/json" in content_type:
            return JSONResponse({"status": True, "prediction": str(prediction)})
        return templates.TemplateResponse(
            request,
            "loanTemplate.html",
            {"context": status},
        )
    except Exception as e:
        if request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"status": False, "error": str(e)}, status_code=500)
        return templates.TemplateResponse(
            request,
            "loanTemplate.html",
            {"context": f"Error occurred: {e}"},
            status_code=500,
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
