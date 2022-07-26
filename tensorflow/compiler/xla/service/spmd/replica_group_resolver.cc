#include "tensorflow/compiler/xla/service/spmd/replica_group_resolver.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace spmd {
 
StatusOr<bool> ReplicaGroupResolver::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* ins : computation->instructions()) {
      if (ins->opcode() == HloOpcode::kAllReduce) {
        auto replica_groups = ins->replica_groups();
        if (replica_groups.size() == 1) {  // size=1 means only one group
          replica_groups.clear();
          // Have to make new instruction and replace, modifying existing does not work
          HloInstruction* new_ins =
                ins->parent()->AddInstruction(HloInstruction::CreateAllReduce(
                    ins->shape(), ins->operands(),
                    ins->to_apply(), replica_groups, 
                    ins->ToProto().constrain_layout(),
                    ins->channel_id(), ins->ToProto().use_global_device_ids()));
          TF_RETURN_IF_ERROR(ins->ReplaceAllUsesWith(new_ins));
          changed = true;
        }
      }
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE(true).Run(module).status());
  }

  return changed;
}
 
}  // namespace spmd
}  // namespace xla