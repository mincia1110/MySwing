"""Debug: check what data the report generation step receives."""
from app.db.session import sync_session_factory
from app.db.models import AnalysisResultTable
import uuid, json

session = sync_session_factory()
try:
    result = session.query(AnalysisResultTable).filter(
        AnalysisResultTable.analysis_id == uuid.UUID("677a75cf-2f88-4695-a19d-24ed55ec8938")
    ).first()
    
    if not result:
        print("No result found")
    else:
        bio = result.biomechanics_data
        phases = result.swing_phases_data
        evals = result.evaluations_data
        
        print(f"=== Biomechanics ===")
        print(f"bat_speed: {bio.get('bat_speed')}")
        print(f"launch_angle: {bio.get('attack_angle')}")
        print(f"hand_path_efficiency: {bio.get('hand_path_efficiency')}")
        print(f"rotation: {bio.get('rotation')}")
        print(f"unmeasurable: {bio.get('unmeasurable_metrics')}")
        
        print(f"\n=== Swing Phases ===")
        print(f"phases: {phases.get('phases', {})}")
        print(f"anomalies: {phases.get('anomalies', [])}")
        
        print(f"\n=== Evaluations ===")
        if isinstance(evals, list):
            print(f"Count: {len(evals)}")
            for e in evals[:3]:
                print(f"  {e.get('metric_name')}: {e.get('measured_value')} ({e.get('rating')})")
        else:
            print(f"Type: {type(evals)}, keys: {evals.keys() if isinstance(evals, dict) else 'N/A'}")
finally:
    session.close()
