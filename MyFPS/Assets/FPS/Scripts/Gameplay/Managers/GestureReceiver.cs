using System;
using System.Threading;
using UnityEngine;
using NetMQ;
using Unity.FPS.Gameplay;   // For PlayerCharacterController and PlayerWeaponsManager
using Unity.FPS.Game;
using NetMQ.Sockets;

public class GestureReceiver : MonoBehaviour
{
    public string connectionAddress = "tcp://localhost:5555";  // Python sender endpoint
    public PlayerCharacterController playerController;
    public PlayerWeaponsManager weaponsManager;

    private Thread listenerThread;
    private bool isRunning = true;
    private readonly object commandLock = new object();
    private string latestCommand = null;

    void Start()
    {
        // Start listening for gesture commands in a background thread
        listenerThread = new Thread(ListenForGestures);
        listenerThread.Start();
    }

    void ListenForGestures()
    {
        AsyncIO.ForceDotNet.Force();
        using (var subscriber = new SubscriberSocket())
        {
            subscriber.Connect(connectionAddress);
            subscriber.Subscribe(""); // Subscribe to all messages
            Debug.Log("GestureReceiver connected to " + connectionAddress);

            while (isRunning)
            {
                try
                {
                    string message = subscriber.ReceiveFrameString();
                    lock (commandLock)
                    {
                        latestCommand = message;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning("GestureReceiver error: " + e.Message);
                }
            }
        }
        NetMQConfig.Cleanup();
    }

    void Update()
    {
        string command = null;

        lock (commandLock)
        {
            if (latestCommand != null)
            {
                command = latestCommand;
                latestCommand = null;
            }
        }

        if (command != null)
        {
            HandleCommand(command);
        }
    }

    //void HandleCommand(string command)
    //{
    //    command = command.ToUpperInvariant();
    //    Debug.Log($"Received Gesture Command: {command}");

    //    switch (command)
    //    {
    //        case "MOVE_FORWARD":
    //            if (playerController != null)
    //            {
    //                Debug.Log("[GestureReceiver] Forcing forward movement!");

    //                // If your player uses CharacterController
    //                var characterController = playerController.GetComponent<CharacterController>();
    //                if (characterController != null)
    //                {
    //                    Vector3 moveDirection = playerController.transform.forward * 5f * Time.deltaTime;
    //                    characterController.Move(moveDirection);
    //                }
    //                else
    //                {
    //                    // If no CharacterController, fallback to transform movement
    //                    playerController.transform.position += playerController.transform.forward * 5f * Time.deltaTime;
    //                }
    //            }
    //            else
    //            {
    //                Debug.LogWarning("[GestureReceiver] PlayerCharacterController not found!");
    //            }
    //            break;

    //        case "MOVE_BACK":
    //            playerController.transform.Translate(Vector3.back * Time.deltaTime * 5f);
    //            break;

    //        case "MOVE_LEFT":
    //            playerController.transform.Translate(Vector3.left * Time.deltaTime * 5f);
    //            break;

    //        case "MOVE_RIGHT":
    //            playerController.transform.Translate(Vector3.right * Time.deltaTime * 5f);
    //            break;

    //        case "JUMP":
    //            if (playerController.IsGrounded)
    //            {
    //                playerController.CharacterVelocity = new Vector3(
    //                    playerController.CharacterVelocity.x,
    //                    0f,
    //                    playerController.CharacterVelocity.z
    //                );
    //                playerController.CharacterVelocity += Vector3.up * playerController.JumpForce;
    //                if (playerController.AudioSource && playerController.JumpSfx)
    //                    playerController.AudioSource.PlayOneShot(playerController.JumpSfx);
    //            }
    //            break;

    //        case "SHOOT":
    //            var activeWeapon = weaponsManager.GetActiveWeapon();
    //            if (activeWeapon != null)
    //            {
    //                // Force the weapon to shoot regardless of input system
    //                if (!activeWeapon.IsReloading && weaponsManager.enabled)
    //                {
    //                    Debug.Log("[GestureReceiver] Forcing weapon fire!");
    //                    activeWeapon.HandleShootInputs(true, true, false);
    //                }
    //                else
    //                {
    //                    Debug.Log("[GestureReceiver] Cannot shoot — weapon is reloading or inactive.");
    //                }
    //            }
    //            else
    //            {
    //                Debug.LogWarning("[GestureReceiver] No active weapon found!");
    //            }
    //            break;

    //        case "RELOAD":
    //            var weapon = weaponsManager.GetActiveWeapon();
    //            if (weapon != null)
    //            {
    //                weapon.StartReloadAnimation(); //correct method for reload
    //            }
    //            break;

    //        default:
    //            Debug.LogWarning("Unknown gesture command: " + command);
    //            break;
    //    }
    //}

    void HandleCommand(string command)
    {
        command = command.ToUpperInvariant();
        Debug.Log($"[GestureReceiver] Received: {command}");

        // Split the command string "GESTURE|MOVEMENT"
        string[] parts = command.Split('|');

        if (parts.Length >= 2)
        {
            string gesture = parts[0];   // e.g., "SHOOT"
            string movement = parts[1];  // e.g., "MOVE_FORWARD"
            float lookX = 0f, lookY = 0f;

            // Parse look deltas if present
            if (parts.Length >= 3)
            {
                string[] lookParts = parts[2].Split(',');
                if (lookParts.Length == 2 &&
                    float.TryParse(lookParts[0], out float lx) &&
                    float.TryParse(lookParts[1], out float ly))
                {
                    lookX = lx;
                    lookY = ly;
                }
            }
            // Apply both commands to the input provider
            //GestureInputProvider.Instance?.ApplyGestureCommand(gesture);
            //GestureInputProvider.Instance?.ApplyGestureCommand(movement);
            // Send everything to the GestureInputProvider
            if (GestureInputProvider.Instance != null)
            {
                GestureInputProvider.Instance.ApplyGestureCommand(gesture);
                GestureInputProvider.Instance.ApplyGestureCommand(movement);
                GestureInputProvider.Instance.LookDelta = new Vector2(lookX, -lookY); // invert Y for natural look
            }
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        listenerThread?.Join();
        NetMQConfig.Cleanup();
    }
}
