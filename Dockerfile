FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pschultzendorff/hcplayground -b reproducable 

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e /app/hcplayground

ENV MPLBACKEND=Agg

CMD ["python", "hcplayground/scripts/buckley_leverett/viscous.py"]
