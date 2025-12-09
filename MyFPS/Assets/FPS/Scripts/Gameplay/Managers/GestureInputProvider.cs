using UnityEngine;

public class GestureInputProvider : MonoBehaviour
{
    public static GestureInputProvider Instance;

    // Movement
    public Vector2 MoveAxis;
    public Vector2 LookDelta;
    public float LookSmoothness = 0.85f; // decay factor for smooth look

    // Action gestures
    public bool Jump;
    public bool Shoot;
    public bool Aim;
    public bool Sprint;
    public bool CrouchDown;
    public bool CrouchUp;
    public bool Reload;

    // Weapon control
    public int SwitchWeaponDirection;
    public int SelectWeaponSlot;

    void Awake()
    {
        Instance = this;
    }

    // Optional: Reset gesture inputs each frame so one-shot actions don't get stuck
    void LateUpdate()
    {
        Jump = false;
        Reload = false;
        CrouchDown = false;
        CrouchUp = false;
        LookDelta *= LookSmoothness;

    }

    // This is called from your NetMQ GestureReceiver
    public void ApplyGestureCommand(string cmd)
    {
        if (string.IsNullOrEmpty(cmd))
        {
            return; // Ignore empty or NONE commands
        }
        cmd = cmd.ToUpperInvariant();
        switch (cmd)
        {
            case "MOVE_FORWARD":
                MoveAxis = Vector2.up;
                Debug.Log("Gesture move forward");
                break;
            case "MOVE_BACKWARD":
                MoveAxis = Vector2.down;
                break;
            case "MOVE_LEFT":
                MoveAxis = Vector2.left;
                break;
            case "MOVE_RIGHT":
                MoveAxis = Vector2.right;
                break;

            case "JUMP":
                Jump = true;
                break;
            case "SHOOT":
                Shoot = true;
                break;
            case "AIM":
                Aim = true;
                break;
            case "SPRINT":
                Sprint = true;
                break;
            case "CROUCH_DOWN":
                CrouchDown = true;
                break;
            case "CROUCH_UP":
                CrouchUp = true;
                break;
            case "RELOAD":
                Reload = true;
                break;
            case "NEXT_WEAPON":
                SwitchWeaponDirection = 1;
                break;
            case "PREV_WEAPON":
                SwitchWeaponDirection = -1;
                break;
            case "STOP":
                Shoot = false; // Turn shooting OFF
                Jump = false;
                break;
            case "NONE":
                MoveAxis = Vector2.zero;
                break;
            default:
                break;
        }
    }
}
