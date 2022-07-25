# 회전 관련 상수
TURN_LIMIT = 0.1
GO_TURN_LIMIT = 0.2
EDGE_TURN = 0.6
DEFAULT_TURN_SPEED = 0.6
DEFAULT_GO_SPEED = 0.5

# 직진 관련 상수
MAX_SPEED = 1.0
MIN_SPEED = 0.2
DEFAULT_SPEED = 0.85
TOO_CLOSE_DIST = 1000
STABLE_MIN_DIST = 2000
STABLE_MAX_DIST = 2500
START_SPEED_DOWN_DIST = 2750
TOO_FAR_DIST = 3000

def drive(cx, frame, turn, speed, person_distance):
    # Target의 위치 파악(좌우 회전이 필요한 지)
    # |<------->|<------------->|<------->|<------->|<------------->|<------->|
    #      TURN_LIMIT      GO_TURN_LIMIT     (1-GO_TURN_LIMIT)(1-TURN_LIMIT)      
    p_cx = cx / frame.shape[1]
    if p_cx <= TURN_LIMIT:
        key = 'turn_left'
        turn = EDGE_TURN
    elif p_cx <= GO_TURN_LIMIT:
        key = 'go_turn_left'
        speed = DEFAULT_GO_SPEED
        turn = DEFAULT_TURN_SPEED
    elif p_cx >= (1 - GO_TURN_LIMIT):
        key = 'go_turn_left'
        speed = DEFAULT_GO_SPEED
        turn = DEFAULT_TURN_SPEED
    elif p_cx >= (1 - TURN_LIMIT):
        key = 'turn_right'
        turn = EDGE_TURN
        
    # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
    # [로봇]|<----정지---->|<----속도 감소---->|<------직진------>|<-----속도 증가----->|<----속도 감소---->|<-----정지----- ...
    #      0      TOO_CLOSE_DIST   STABLE_MIN_DIST   STABLE_MAX_DIST   START_SPEED_DOWN_DIST  TOO_FAR_DIST
    else:
        if person_distance <= TOO_CLOSE_DIST: 
            key = 'stop'
        elif person_distance < STABLE_MAX_DIST:
            if speed > MIN_SPEED:
                key = 'linear_speed_down'
            else:
                key = 'go'
                speed = MIN_SPEED
        elif person_distance <= STABLE_MAX_DIST:
            key = 'go'
            speed = DEFAULT_SPEED
        elif person_distance <= START_SPEED_DOWN_DIST:
            if speed < MAX_SPEED:
                speed = 'linear_speed_up'
            else:
                key = 'go'
                speed = MAX_SPEED
        elif person_distance < TOO_FAR_DIST:
            if speed > MIN_SPEED:
                key = 'linear_speed_down'
            else:
                key = 'go'
                speed = MIN_SPEED
        else: key = 'stop'
        
    return key, speed, turn