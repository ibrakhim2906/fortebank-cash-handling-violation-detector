"""Tri-state observations. A detector miss is unknown, never proven absence."""
from service.pipeline.tracker import iou

NAMES=('staff_present','cashier_at_desk','client_present','extra_person','person_entered','cashier_leaving',
       'safe_open','desk_items','phone_in_hand','bag_on_desk','drink_on_desk','outsider_present',
       'door_open','person_in_doorway','cash_on_desk','cashier_at_safe','key_in_safe_lock',
       'laptop_screen_on','hand_to_hand','counter_active','client_left','cash_received','cash_given',
       'id_document_shown','document_given','drink_at_face_or_desk','book_in_hands')
CFG=dict(handheld_overlap=.03,max_observation_gap_sec=.6)


def timeline(observations,scene):
    rows=[]
    for row in observations['frames']:
        facts=dict.fromkeys(NAMES);evidence={};detections=row['detections']
        people=[d for d in detections if d['track_id'] in row['foreground_person_tracks']]
        facts['n_people_observed']=len(people)  # lower bound, not room occupancy
        staff=next((d for d in people if d['track_id']==scene.get('cashier_track')),None)
        if staff:
            facts['staff_candidate_observed']=True;evidence['staff_candidate_observed']=[staff['box_xyxy']]
            for kind,name in [(67,'phone_near_staff'),(73,'book_near_staff')]:
                boxes=[d['box_xyxy'] for d in detections if d['class_id']==kind and iou(d['box_xyxy'],staff['box_xyxy'])>=CFG['handheld_overlap']]
                if boxes:facts[name]=True;evidence[name]=boxes
        # Objects, hand proximity, screen brightness, empty detections and pixel
        # change alone deliberately do not certify any compliance predicate.
        rows.append(dict(timestamp_sec=row['timestamp_sec'],source_frame_index=row['source_frame_index'],facts=facts,evidence=evidence))
    return rows
