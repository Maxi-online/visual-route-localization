import json
from app.vloc.geo import ll_to_enu, enu_to_ll
from app.vloc.confidence import ConfidenceEstimator
from app.vloc.smoothing import EwmaSmoother

def run():
    origin=(55.0,37.0);query=(55.0001,37.0001)
    point=ll_to_enu(*query,*origin);restored=enu_to_ll(point,*origin)
    confidence=ConfidenceEstimator().compute_visual(0.8,0.6,1.0,30)
    smoother=EwmaSmoother();smoother.update(*origin,confidence);smoothed=smoother.update(*query,confidence)
    return {"mode":"synthetic-geodesy-confidence-smoothing","enu_m":{"east":point.east_m,"north":point.north_m},"round_trip":restored,"confidence":confidence,"smoothed":smoothed,"visual_model_executed":False}
if __name__=="__main__":print(json.dumps(run(),indent=2))
