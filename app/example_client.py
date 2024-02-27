import zmq

import maestro_api.maestro_messages as mm


# ctx = zmq.Context.instance()
# client = ctx.socket(zmq.REQ)
# client.connect('tcp://einstein.dhcp.lbl.gov:9000')

msg = mm.MaestroLVStartupMessage(
            AI_Controller = 'Aardvark',
            # AIModeparms=[
            #     AIModeparm(device_name="motors::X", enabled_=True, low=0, high=1, min_step=0.01),
            #     AIModeparm(device_name="motors::Y", enabled_=True, low=0, high=1, min_step=0.01)
            # ],
            AIModeparms=[
                mm.AIModeparm(device_name="motors::X", enabled_=True, low=0, high=1, min_step=0.1),
                mm.AIModeparm(device_name="motors::Y", enabled_=True, low=-1, high=2, min_step=0.2)
            ],
            # This is the number of AI cycles to go through
            max_count=12,
            method="initialize",
            scan_descriptors=mm.ScanDescriptors(
                Scan_Descriptor=[
                    mm.ScanDescriptorItem(num_positions=89,
                                        Offsets=[0,0],
                                        Range=[mm.RangeItem(End=0,N=1,Start=0)],
                                        Scan_Type="Computed",
                                        Tab_Posns_=[[0.0,0.1]],
                                        device_descriptor=mm.DeviceDescriptor(
                                            NEXUS_Path_Class_rel_to_entry="sample",
                                            device_name="None",
                                            subdevices=[
                                                mm.Subdevice(hi=0,lo=0,name="null",parms=[],units="")
                                            ]))
                ],
                Scan_Devices_in_Parallel=False,
                # This is how many frames per cycle *I think*
                total_num_cycles=1,
            )
        )

# msg = mm.MaestroLVCloseMessage()

# client.send(msg.model().encode('utf-8'))
# msg = client.recv()
print(msg.model_dump_json(by_alias=True))
