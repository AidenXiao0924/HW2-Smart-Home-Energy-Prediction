"""Generated Draft 2020-12 schemas plus semantic and public-safety checks."""
from datetime import datetime
import math
import re
from jsonschema import Draft202012Validator
from src.export.reference import FEATURES, ordered_values, validate_artifact


def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite JSON number')
    if isinstance(value,dict):
        for item in value.values():finite_tree(item)
    if isinstance(value,list):
        for item in value:finite_tree(item)


def public_safe(value):
    finite_tree(value)
    if isinstance(value,str):
        if re.search(r'(?i)((?<![a-z0-9])[a-z]:[\\/]|file://|/users/|/home/|\\\\|bearer\s+|sk-[a-z0-9]{12,}|gh[pousr]_[a-z0-9]+|access_token=)',value):
            raise ValueError('Local path or credential-shaped text in public artifact')
    elif isinstance(value,dict):
        for key,item in value.items():
            if re.search(r'(?i)(password|secret|api.?key|token)',key):
                raise ValueError('Credential field in public artifact')
            public_safe(item)
    elif isinstance(value,list):
        for item in value:public_safe(item)


def obj(properties,required=None,extra=False):
    return dict(type='object',properties=properties,required=list(properties) if required is None else required,
                additionalProperties=extra)


def schemas():
    number=dict(type='number')
    string=dict(type='string')
    sha=dict(type='string',pattern='^[a-f0-9]{64}$')
    source=obj(dict(path=dict(type='string',pattern='^(reports|artifacts|configs)/'),sha256=sha))
    splits=['ALL_RAW','TRAIN_CV','VALIDATION','TRAIN_VALIDATION','CALIBRATION','TEST']
    block=obj(dict(split=dict(enum=splits),source=source,transformation=string,values=dict(type='object')))
    summary=obj(dict(schema_version=dict(const='1.0.0'),model_version=dict(const='mlr-1.0.0'),
        dataset=block,splits=block,feature_selection=block,training_cv=block,validation=block,
        final_model=block,calibration=block,test=block,diagnostics=block,training_vif=block,
        attribution=dict(type='object'),limitations=dict(type='array',items=string),
        units=obj(dict(mae=dict(const='Wh'),mse=dict(const='Wh^2'),rmse=dict(const='Wh'),
                       interval_width=dict(const='Wh'),r2=dict(const='dimensionless'),coverage=dict(const='fraction'))),
        lazy_chart_files=dict(type='array',items=string)))
    date=dict(type='string',pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    row_properties={f:dict(type=['number','null']) for f in FEATURES[:-1]}
    row_properties.update(date=date,Hour=dict(type='integer',minimum=0,maximum=23),
                          Appliances=dict(type='number',minimum=0))
    row=obj(row_properties,required=[*FEATURES[:-1],'date'],extra=True)
    error=obj(dict(schema_version=dict(const='1.0.0'),status=dict(const='INVALID_INPUT'),
        errors=dict(type='array',minItems=1,items=obj(dict(
            code=dict(enum=['MISSING_COLUMN','INVALID_DATE','NONFINITE_OR_TYPE','TIME_FEATURE_MISMATCH','INVALID_TARGET',
                            'FILE_TOO_LARGE','TOO_MANY_ROWS','UNSUPPORTED_URL','CSV_PARSE_ERROR','MODEL_VERSION_MISMATCH']),
            row=dict(type=['integer','null'],minimum=1),field=dict(type=['string','null']),message=string)))))
    chart_point=obj(dict(timestamp=date,count=dict(type='integer',minimum=1),
        actual_mean=number,actual_max=number,prediction_mean=number,lower_mean=number,upper_mean=number,
        covered_count=dict(type='integer',minimum=0)))
    series=obj(dict(schema_version=dict(const='1.0.0'),model_version=dict(const='mlr-1.0.0'),
        split=dict(const='TEST'),source=source,aggregation=dict(const='source-local hourly; arithmetic means, actual maxima and exact counts'),
        source_rows=dict(const=2962),rows=dict(type='array',maxItems=500,items=chart_point)))
    scatter_point=obj(dict(timestamp=date,actual=number,prediction=number,residual=number,
                           lower=number,upper=number,interval_valid=dict(type='boolean'),warnings=string))
    residual=obj(dict(schema_version=dict(const='1.0.0'),model_version=dict(const='mlr-1.0.0'),split=dict(const='TEST'),
        source=source,sampling=string,source_rows=dict(const=2962),scatter=dict(type='array',maxItems=512,items=scatter_point),
        histogram=obj(dict(edges=dict(type='array',items=number),counts=dict(type='array',items=dict(type='integer',minimum=0)))),
        qq=obj(dict(theoretical=dict(type='array',items=number),ordered_residual=dict(type='array',items=number))),
        residual_acf=dict(type='array',items=number)))
    cvrow=obj(dict(experiment_id=string,candidate_set=dict(enum=['deployable27','research28']),
        method=dict(enum=['select_k_best','rfe','forward_sfs','lasso','permutation']),k=dict(type='string'),
        effective_k=dict(type='integer'),cv_rmse_mean=number,cv_rmse_std=number,feature_jaccard_mean=number,
        compute_seconds=number,classification=string))
    cv=obj(dict(schema_version=dict(const='1.0.0'),model_version=dict(const='mlr-1.0.0'),split=dict(const='TRAIN_CV'),
        source=source,aggregation=dict(const='Five chronological outer folds; RMSE population SD (ddof=0)'),
        configurations=dict(type='array',minItems=50,maxItems=50,items=cvrow)))
    fixtures=obj(dict(schema_version=dict(const='1.0.0'),model_version=dict(const='mlr-1.0.0'),
        source=source,source_model_sha256=sha,public_model_sha256=sha,tolerance=string,javascript_execution=dict(const='NOT_RUN'),
        fixtures=dict(type='array',minItems=10,items=obj(dict(kind=string,input=row,expected=obj(dict(
            predicted=number,lower=number,upper=number,interval_valid=dict(type='boolean'),warnings=dict(type='array',items=string)))))),
        interval_boundary_cases=dict(type='array',minItems=2,items=dict(type='object')),
        invalid_input_cases=dict(type='array',minItems=4,items=dict(type='object'))))
    result={'research-summary':summary,'external-row':row,'external-error':error,'test-series':series,
            'test-residuals':residual,'training-cv':cv,'parity-fixtures':fixtures}
    for name,value in result.items():
        value.update({'$schema':'https://json-schema.org/draft/2020-12/schema','$id':f'urn:smart-home-energy:{name}:1.0.0'})
    return result


def validate(value,schema):
    finite_tree(value)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)


def external_row_errors(row, row_number=1):
    errors=[]
    def add(code,field,message):errors.append(dict(code=code,row=row_number,field=field,message=message))
    if not isinstance(row,dict):
        add('CSV_PARSE_ERROR',None,'Parsed row must be a named object')
    else:
        for f in [*FEATURES[:-1],'date']:
            if f not in row:add('MISSING_COLUMN',f,'Required column absent')
        if not errors:
            try:ordered_values(row)
            except (ValueError,TypeError):
                date=row.get('date')
                try:
                    if not isinstance(date,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',date):raise ValueError()
                    dt=datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
                except (TypeError,ValueError):add('INVALID_DATE','date','Use a valid source-local YYYY-MM-DD HH:mm:ss date')
                else:
                    if 'Hour' in row and (isinstance(row['Hour'],bool) or not isinstance(row['Hour'],(int,float)) or row['Hour']!=dt.hour):
                        add('TIME_FEATURE_MISMATCH','Hour','Hour must agree with source-local date')
                    for f in FEATURES[:-1]:
                        v=row[f]
                        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)):
                            add('NONFINITE_OR_TYPE',f,'Finite numeric value or null required')
        if 'Appliances' in row:
            v=row['Appliances']
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                add('INVALID_TARGET','Appliances','Optional target must be finite and nonnegative')
    if errors:return dict(schema_version='1.0.0',status='INVALID_INPUT',errors=errors)
    validate(row,schemas()['external-row'])
    return None
